# mypy: allow-untyped-defs

import functools
import inspect
import itertools
import warnings
import weakref
from collections import namedtuple, OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Optional, overload, TypeVar, Union
from typing_extensions import Self

import torch
from torch import device, dtype, Tensor
from torch._prims_common import DeviceLikeType
from torch.nn.parameter import Buffer, Parameter
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.hooks import BackwardHook, RemovableHandle


# 定义模块公开接口列表（控制from module import *时暴露的内容）
__all__ = [
    "register_module_forward_pre_hook",      # 注册前向传播预处理钩子
    "register_module_forward_hook",          # 注册前向传播后处理钩子
    "register_module_full_backward_pre_hook",# 注册完整反向传播预处理钩子
    "register_module_backward_hook",         # 注册反向传播钩子(已废弃，保留兼容)
    "register_module_full_backward_hook",    # 注册完整反向传播后处理钩子
    "register_module_buffer_registration_hook",  # 注册buffer添加时的钩子
    "register_module_module_registration_hook",  # 注册子模块添加时的钩子
    "register_module_parameter_registration_hook",# 注册参数添加时的钩子
    "Module",                               # 基础模块类
]

# 梯度类型注解：可以是张量或张量元组
_grad_t = Union[tuple[Tensor, ...], Tensor]

# 类型变量T，用于注解返回self的方法，确保子类方法返回正确的类型
# 详见MyPy文档关于泛型方法的说明
T = TypeVar("T", bound="Module")


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    """用于记录模型状态字典加载时的键不匹配情况"""
    __slots__ = ()  # 禁止动态属性创建以节省内存

    def __repr__(self):
        """自定义输出格式：
        - 当没有缺失或意外键时显示成功消息
        - 否则调用父类的元组表示法
        """
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    # 使str()和repr()输出一致
    __str__ = __repr__

def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: dict[int, Callable] = OrderedDict()


class _WrappedHook:
    """包装钩子函数的类，用于处理普通函数或与模块关联的钩子函数"""

    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        """初始化包装的钩子
        Args:
            hook: 要包装的钩子函数
            module: 可选的关联模块，如果提供则创建弱引用
        """
        self.hook: Callable = hook  # 原始钩子函数
        functools.update_wrapper(self, hook)  # 保留原始函数的元数据

        self.with_module: bool = False  # 标记是否关联模块

        if module is not None:
            # 创建模块的弱引用，避免循环引用导致内存泄漏
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """调用钩子函数"""
        if self.with_module:
            # 如果关联模块，先获取模块引用
            module = self.module()
            if module is None:
                raise RuntimeError("尝试调用已销毁模块的钩子！")
            # 将模块作为第一个参数传入
            return self.hook(module, *args, **kwargs)
        # 普通钩子直接调用
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> dict:
        """序列化时调用的方法"""
        result = {
            "hook": self.hook, 
            "with_module": self.with_module
        }
        if self.with_module:
            # 如果关联模块，保存模块的实际对象（不是弱引用）
            result["module"] = self.module()
        return result

    def __setstate__(self, state: dict):
        """反序列化时调用的方法"""
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError("尝试恢复已销毁模块的钩子！")
            # 重新创建模块的弱引用
            self.module = weakref.ref(state["module"])

r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_backward_hooks: dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks: dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: dict[int, bool] = OrderedDict()
_global_forward_hooks_with_kwargs: dict[int, bool] = OrderedDict()

_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


def register_module_buffer_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""注册一个适用于所有模块的buffer注册全局钩子函数

    .. warning ::
        注意：这会向`nn.Module`模块添加全局状态

    每次调用:func:`register_buffer`时都会触发该钩子。
    钩子函数应遵循以下签名格式::

        hook(module, name, buffer) -> None 或 返回修改后的新buffer

    钩子可以修改输入参数，或者返回单个修改后的值。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            返回一个可移除句柄，通过调用``handle.remove()``可以移除该钩子
    """
    # 创建可移除句柄并将其添加到全局buffer注册钩子字典
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""注册一个适用于所有模块的子模块注册全局钩子函数

    .. warning ::
        注意：这会向`nn.Module`模块添加全局状态

    每次调用:func:`register_module`时都会触发该钩子。
    钩子函数应遵循以下签名格式::

        hook(module, name, submodule) -> None 或 返回修改后的新子模块

    钩子可以修改输入参数，或者返回单个修改后的值。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            返回一个可移除句柄，通过调用``handle.remove()``可以移除该钩子
    """
    # 创建可移除句柄并将其添加到全局模块注册钩子字典
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle

def register_module_parameter_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""注册一个适用于所有模块的参数注册全局钩子函数

    .. warning ::
        警告：这会向`nn.Module`模块添加全局状态

    每次调用:func:`register_parameter`时都会触发该钩子。
    钩子函数应遵循以下签名格式::
        hook(module, name, param) -> None 或 返回修改后的新参数

    钩子可以修改输入参数，或者返回单个修改后的值。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            返回一个可移除句柄，通过调用``handle.remove()``可以移除该钩子
    """
    # 创建可移除句柄并添加到全局参数注册钩子字典
    handle = RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""注册一个适用于所有模块的前向传播预处理全局钩子

    .. warning ::
        警告：这会向`nn.module`模块添加全局状态，
        且仅用于调试/性能分析目的

    每次调用:func:`forward`前都会触发该钩子。
    钩子函数应遵循以下签名格式::
        hook(module, input) -> None 或 修改后的输入

    输入仅包含传给模块的位置参数，关键字参数不会传给钩子，
    只会传给``forward``方法。钩子可以修改输入参数，用户可以返回
    一个元组或单个修改值。如果返回单个值(非元组)，我们会自动将其包装成元组。

    此钩子的优先级高于通过``register_forward_pre_hook``注册的模块特定钩子。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            返回一个可移除句柄，通过调用``handle.remove()``可以移除该钩子
    """
    # 创建可移除句柄并添加到全局前向预处理钩子字典
    handle = RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(
    hook: Callable[..., None],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    r"""为所有模块注册全局前向传播钩子

    .. warning ::
        警告：这会向`nn.module`模块添加全局状态，
        且仅用于调试/性能分析目的

    每次调用:func:`forward`计算出输出后都会触发该钩子。
    钩子函数应遵循以下签名格式::
        hook(module, input, output) -> None 或 修改后的输出

    输入仅包含传给模块的位置参数，关键字参数不会传给钩子，
    只会传给``forward``方法。可以通过返回新值来修改模块的输出，
    该值将替换:func:`forward`函数的原始输出。

    参数:
        hook (Callable): 要注册的用户定义钩子
        with_kwargs (bool): 是否传递关键字参数给钩子
        always_call (bool): 如果为``True``，即使模块调用时发生异常也会执行钩子
            默认: ``False``
    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            返回一个可移除句柄，通过调用``handle.remove()``可以移除该钩子

    此钩子会在通过``register_forward_hook``注册的模块特定钩子之前执行。
    """
    # 创建可移除句柄并管理多个相关字典
    handle = RemovableHandle(
        _global_forward_hooks, 
        extra_dict=_global_forward_hooks_always_called
    )
    _global_forward_hooks[handle.id] = hook
    if with_kwargs:
        _global_forward_hooks_with_kwargs[handle.id] = True
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle
def register_module_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""为所有模块注册全局反向传播钩子(已弃用)

    注意：此函数已被弃用，推荐使用
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    未来版本中此函数的行为可能会改变。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            可移除句柄，调用``handle.remove()``可删除此钩子

    """
    global _global_is_full_backward_hook
    # 检查是否已存在完整反向钩子的冲突
    if _global_is_full_backward_hook is True:
        raise RuntimeError(
            "不能同时使用常规反向钩子和完整反向钩子作为全局模块钩子。"
            "请只选择其中一种使用。"
        )

    _global_is_full_backward_hook = False  # 标记为常规反向钩子

    # 创建可移除句柄并添加到全局反向钩子字典
    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_full_backward_pre_hook(
    hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""为所有模块注册全局完整反向传播预处理钩子

    .. warning ::
        警告：这会向`nn.module`模块添加全局状态，
        且仅用于调试/性能分析目的

    使用此函数注册的钩子行为与
    :meth:`torch.nn.Module.register_full_backward_pre_hook`注册的钩子相同。
    详见其文档说明。

    此钩子会在通过:meth:`torch.nn.Module.register_full_backward_pre_hook`
    注册的模块特定钩子之前执行。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            可移除句柄，调用``handle.remove()``可删除此钩子
    """
    # 创建可移除句柄并添加到全局反向预处理钩子字典
    handle = RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle


def register_module_full_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""为所有模块注册全局完整反向传播钩子

    .. warning ::
        警告：这会向`nn.module`模块添加全局状态，
        且仅用于调试/性能分析目的

    使用此函数注册的钩子行为与
    :meth:`torch.nn.Module.register_full_backward_hook`注册的钩子相同。
    详见其文档说明。

    此钩子会在通过:meth:`torch.nn.Module.register_full_backward_hook`
    注册的模块特定钩子之前执行。

    返回:
        :class:`torch.utils.hooks.RemovableHandle`:
            可移除句柄，调用``handle.remove()``可删除此钩子
    """
    global _global_is_full_backward_hook
    # 检查是否已存在常规反向钩子的冲突
    if _global_is_full_backward_hook is False:
        raise RuntimeError(
            "不能同时使用常规反向钩子和完整反向钩子作为全局模块钩子。"
            "请只选择其中一种使用。"
        )

    _global_is_full_backward_hook = True  # 标记为完整反向钩子

    # 创建可移除句柄并添加到全局反向钩子字典
    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle

# 通过将forward定义为值而非函数来绕过mypy的逆变规则检查
# 另见: https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""定义每次调用时执行的计算操作。

    所有子类都应该重写此方法。

    .. note::
        虽然前向传递的逻辑需要在此函数中定义，
        但实际调用时应该使用模块实例(如model(x))而非直接调用此方法，
        因为前者会执行已注册的钩子而后者会静默忽略它们。

    此方法故意抛出未实现错误，强制子类必须实现forward方法。
    """
    raise NotImplementedError(
        f'模块 [{type(self).__name__}] 缺少必需的 "forward" 方法实现'
    )

class Module:
    r"""所有神经网络模块的基类。

    你的模型应该继承这个类。

    模块可以包含其他模块，形成树状嵌套结构。你可以像常规属性一样分配子模块::
    
        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()  # 必须首先调用父类初始化
                self.conv1 = nn.Conv2d(1, 20, 5)  # 子模块会自动注册
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    这样分配的子模块会被自动注册，当调用:meth:`to`等方法时，它们的参数也会被转换。

    .. note::
        如上例所示，必须在子类属性分配前调用父类的``__init__()``方法。

    :ivar training: 布尔值，表示此模块处于训练还是评估模式。
    :vartype training: bool
    """

    dump_patches: bool = False  # 用于序列化的补丁标记

    _version: int = 1
    r"""该版本号支持:meth:`load_state_dict`的更好向后兼容。
    在:meth:`state_dict`中，版本号会保存在返回状态字典的`_metadata`属性中并被序列化。
    `_metadata`是一个字典，其键遵循状态字典的命名规范。
    参见``_load_from_state_dict``了解如何在加载时使用此信息。

    如果模块添加/删除了新参数/缓冲区，应该增加此版本号，
    模块的`_load_from_state_dict`方法可以比较版本号，
    如果状态字典来自变更前，可以做出适当的调整。"""

    # 模块核心状态变量
    training: bool  # 训练/评估模式标志
    _parameters: dict[str, Optional[Parameter]]  # 存储所有参数
    _buffers: dict[str, Optional[Tensor]]  # 存储所有缓冲区
    _non_persistent_buffers_set: set[str]  # 不持久化的缓冲区集合
    
    # 钩子相关存储
    _backward_pre_hooks: dict[int, Callable]  # 反向传播预处理钩子
    _backward_hooks: dict[int, Callable]  # 反向传播钩子
    _is_full_backward_hook: Optional[bool]  # 是否使用完整反向钩子
    
    _forward_hooks: dict[int, Callable]  # 前向传播钩子
    _forward_hooks_with_kwargs: dict[int, bool]  # 接受kwargs的前向钩子标记
    _forward_hooks_always_called: dict[int, bool]  # 异常时也调用的前向钩子
    
    _forward_pre_hooks: dict[int, Callable]  # 前向传播预处理钩子
    _forward_pre_hooks_with_kwargs: dict[int, bool]  # 接受kwargs的前向预处理钩子
    
    # 状态字典相关钩子
    _state_dict_hooks: dict[int, Callable]  # 状态字典钩子
    _load_state_dict_pre_hooks: dict[int, Callable]  # 加载状态字典预处理钩子
    _state_dict_pre_hooks: dict[int, Callable]  # 状态字典预处理钩子
    _load_state_dict_post_hooks: dict[int, Callable]  # 加载状态字典后处理钩子
    
    _modules: dict[str, Optional["Module"]]  # 子模块字典
    
    # 其他功能标志
    call_super_init: bool = False  # 是否调用父类初始化标志
    _compiled_call_impl: Optional[Callable] = None  # 编译后的调用实现
    
def __init__(self, *args, **kwargs) -> None:
    """初始化Module内部状态，同时被nn.Module和ScriptModule共享"""
    # 记录API使用情况（仅记录一次）
    torch._C._log_api_usage_once("python.nn_module")

    # 向后兼容性处理：当call_super_init=False时不允许任何参数
    if self.call_super_init is False and bool(kwargs):
        raise TypeError(
            f"{type(self).__name__}.__init__() 收到了意外的关键字参数 '{next(iter(kwargs))}'"
        )

    if self.call_super_init is False and bool(args):
        raise TypeError(
            f"{type(self).__name__}.__init__() 需要1个位置参数但收到了 {len(args) + 1} 个"
        )

    """
    使用super().__setattr__()而不是直接属性赋值来避免Module.__setattr__的开销。
    Module的__setattr__对参数、子模块和缓冲区有特殊处理，对其他属性则直接调用super().__setattr__。
    """
    # 初始化所有模块内部状态
    super().__setattr__("training", True)  # 默认训练模式
    super().__setattr__("_parameters", {})  # 参数字典
    super().__setattr__("_buffers", {})  # 缓冲区字典
    super().__setattr__("_non_persistent_buffers_set", set())  # 非持久化缓冲区集合
    
    # 初始化各种钩子容器（使用OrderedDict保持顺序）
    super().__setattr__("_backward_pre_hooks", OrderedDict())  # 反向预处理钩子
    super().__setattr__("_backward_hooks", OrderedDict())  # 反向钩子
    super().__setattr__("_is_full_backward_hook", None)  # 完整反向钩子标记
    
    super().__setattr__("_forward_hooks", OrderedDict())  # 前向钩子
    super().__setattr__("_forward_hooks_with_kwargs", OrderedDict())  # 带kwargs的前向钩子
    super().__setattr__("_forward_hooks_always_called", OrderedDict())  # 总是调用的前向钩子
    
    super().__setattr__("_forward_pre_hooks", OrderedDict())  # 前向预处理钩子
    super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())  # 带kwargs的前向预处理钩子
    
    super().__setattr__("_state_dict_hooks", OrderedDict())  # 状态字典钩子
    super().__setattr__("_state_dict_pre_hooks", OrderedDict())  # 状态字典预处理钩子
    super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())  # 加载状态字典预处理钩子
    super().__setattr__("_load_state_dict_post_hooks", OrderedDict())  # 加载状态字典后处理钩子
    
    super().__setattr__("_modules", {})  # 子模块字典

    # 如果设置了call_super_init标志，调用父类初始化
    if self.call_super_init:
        super().__init__(*args, **kwargs)

    # 将forward设置为未实现的默认方法
    forward: Callable[..., Any] = _forward_unimplemented

    def register_buffer(
            self, name: str, tensor: Optional[Tensor], persistent: bool = True
        ) -> None:
        r"""向模块注册缓冲区(buffer)。

        通常用于注册不应被视为模型参数的缓冲区。例如BatchNorm的`running_mean`不是参数，
        但是模块状态的一部分。缓冲区默认是持久化的(persistent)，会与参数一起保存。
        可以通过设置`persistent=False`来改变此行为。持久化和非持久化缓冲区的唯一区别是
        后者不会包含在模块的`state_dict`中。

        缓冲区可以通过给定的名称作为属性访问。

        参数:
            name (str): 缓冲区名称。可以通过此名称从模块访问缓冲区
            tensor (Tensor or None): 要注册的缓冲区。如果是None，则对缓冲区的操作
                (如`cuda`)会被忽略。如果是None，缓冲区**不会**包含在模块的`state_dict`中
            persistent (bool): 缓冲区是否是模块`state_dict`的一部分

        示例::
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """
        # 检查ScriptModule不支持非持久化缓冲区
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule不支持非持久化缓冲区")

        # 参数有效性检查
        if "_buffers" not in self.__dict__:
            raise AttributeError("必须在Module.__init__()调用后才能注册缓冲区")
        elif not isinstance(name, str):
            raise TypeError(f"缓冲区名称必须是字符串，得到 {torch.typename(name)}")
        elif "." in name:
            raise KeyError('缓冲区名称不能包含"."')
        elif name == "":
            raise KeyError('缓冲区名称不能为空字符串""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"属性'{name}'已存在")
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"不能将'{torch.typename(tensor)}'对象赋值给缓冲区'{name}'"
                "(需要torch.Tensor或None)"
            )
        
        # 执行全局缓冲区注册钩子
        for hook in _global_buffer_registration_hooks.values():
            output = hook(self, name, tensor)
            if output is not None:
                tensor = output
        
        # 注册缓冲区
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)  # 从非持久化集合中移除
        else:
            self._non_persistent_buffers_set.add(name)  # 添加到非持久化集合

        def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
            r"""向模块注册参数。

            参数可以通过给定的名称作为属性访问。

            参数:
                name (str): 参数名称。可以通过此名称从模块访问参数
                param (Parameter or None): 要添加到模块的参数。如果是None，
                    则对参数的操作(如`cuda`)会被忽略。如果是None，参数**不会**
                    包含在模块的`state_dict`中。
            """
            # 参数有效性检查
            if "_parameters" not in self.__dict__:
                raise AttributeError("必须在Module.__init__()调用后才能注册参数")
            elif not isinstance(name, str):
                raise TypeError(f"参数名称必须是字符串，得到 {torch.typename(name)}")
            elif "." in name:
                raise KeyError('参数名称不能包含"."')
            elif name == "":
                raise KeyError('参数名称不能为空字符串""')
            elif hasattr(self, name) and name not in self._parameters:
                raise KeyError(f"属性'{name}'已存在")

            if param is None:
                self._parameters[name] = None
            elif not isinstance(param, Parameter):
                raise TypeError(
                    f"不能将'{torch.typename(param)}'对象赋值给参数'{name}'"
                    "(需要torch.nn.Parameter或None)"
                )
            elif param.grad_fn:
                raise ValueError(
                    f"不能将非叶子Tensor赋值给参数'{name}'。模型参数必须显式创建。"
                    f"如果要将'{name}'表示为另一个Tensor的函数，请在forward()方法中计算其值。"
                )
            else:
                # 执行全局参数注册钩子
                for hook in _global_parameter_registration_hooks.values():
                    output = hook(self, name, param)
                    if output is not None:
                        param = output
                self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        r"""向当前模块添加子模块
        
        可以通过给定名称作为属性访问该模块
        
        参数:
            name (str): 子模块名称。可以通过该名称从当前模块访问子模块
            module (Module): 要添加到模块的子模块
        """
        # 检查模块类型有效性
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{torch.typename(module)} 不是Module子类")
        # 检查名称类型有效性
        elif not isinstance(name, str):
            raise TypeError(f"模块名称应为字符串，实际得到 {torch.typename(name)}")
        # 检查名称是否已被占用
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"属性 '{name}' 已存在")
        # 检查名称格式有效性
        elif "." in name:
            raise KeyError(f'模块名称不能包含"."，实际得到: {name}')
        elif name == "":
            raise KeyError('模块名称不能为空字符串""')
        
        # 执行全局模块注册钩子
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        
        # 将模块添加到内部字典
        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""add_module的别名方法"""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """根据target路径获取子模块，不存在则抛出错误
        
        示例:
            对于模块结构A(net_b(net_c(conv), linear))，
            获取linear子模块: get_submodule("net_b.linear")
            获取conv子模块: get_submodule("net_b.net_c.conv")
            
        参数:
            target: 子模块的完整路径字符串
            
        返回:
            torch.nn.Module: target对应的子模块
            
        异常:
            AttributeError: 当路径中任何部分不存在或不是nn.Module实例时抛出
        """
        if target == "":
            return self
        
        # 分割路径为原子部分
        atoms: list[str] = target.split(".")
        mod: torch.nn.Module = self
        
        # 逐级查找子模块
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " 没有属性 `" + item + "`")
            
            mod = getattr(mod, item)
            
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` 不是nn.Module")
        
        return mod


    def set_submodule(
        self, target: str, module: "Module", strict: bool = False
    ) -> None:
        """
        设置或替换指定路径的子模块
        
        参数:
            target: 子模块的完整路径字符串（如"net_b.net_c.conv"）
            module: 要设置的模块对象
            strict: 严格模式标志。False时允许创建新子模块，True时仅允许替换现有子模块
            
        异常:
            ValueError: 当target为空或module不是nn.Module实例时抛出
            AttributeError: 当路径不存在或路径对象不是nn.Module时抛出
        """
        # 检查target非空
        if target == "":
            raise ValueError("必须提供目标子模块名称！")
        
        # 分割路径为原子部分
        atoms: list[str] = target.split(".")
        # 验证模块类型
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"`module`不是nn.Module类型，实际类型为{type(module)}"
            )
        
        # 获取父模块
        if len(atoms) == 1:  # 目标为直接子模块
            parent: torch.nn.Module = self
        else:  # 目标为嵌套子模块
            parent_key = ".".join(atoms[:-1])
            parent = self.get_submodule(parent_key)
        
        # 严格模式检查
        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(
                f"{parent._get_name()} 没有属性 `{atoms[-1]}`"
            )
        
        # 检查现有属性类型
        if hasattr(parent, atoms[-1]):
            mod = getattr(parent, atoms[-1])
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError(f"`{atoms[-1]}` 不是nn.Module")
        
        # 设置新模块
        setattr(parent, atoms[-1], module)
    def get_parameter(self, target: str) -> "Parameter":
        """根据路径字符串获取模块参数
        
        参数:
            target: 参数的完整路径字符串（如"net.conv.weight"）
            
        返回:
            torch.nn.Parameter: 目标参数对象
            
        异常:
            AttributeError: 当路径无效或目标不是Parameter时抛出
        """
        # 分割路径为模块路径和参数名
        module_path, _, param_name = target.rpartition(".")
        
        # 获取参数所在模块
        mod: torch.nn.Module = self.get_submodule(module_path)
        
        # 检查参数是否存在
        if not hasattr(mod, param_name):
            raise AttributeError(
                f"{mod._get_name()} 没有属性 `{param_name}`"
            )
        
        # 获取参数对象
        param: torch.nn.Parameter = getattr(mod, param_name)
        
        # 验证参数类型
        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError(f"`{param_name}` 不是nn.Parameter类型")
        
        return param


    def get_buffer(self, target: str) -> "Tensor":
        """根据给定的target返回对应的buffer，如果不存在则抛出错误。

        关于此方法功能的更详细说明以及如何正确指定target，
        请参阅``get_submodule``的文档字符串。

        参数:
            target: 要查找的buffer的完全限定字符串名称。
                （关于如何指定完全限定字符串，请参考``get_submodule``）

        返回:
            torch.Tensor: target引用的buffer

        抛出:
            AttributeError: 如果target字符串引用了一个无效路径，
                或者解析到的对象不是一个buffer
        """
        # 从target字符串中分割出模块路径和buffer名称
        module_path, _, buffer_name = target.rpartition(".")

        # 获取对应的子模块
        mod: torch.nn.Module = self.get_submodule(module_path)

        # 检查模块是否有该属性
        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " 没有属性 `" + buffer_name + "`"
            )

        # 获取属性值
        buffer: torch.Tensor = getattr(mod, buffer_name)

        # 验证该属性确实是buffer
        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` 不是一个buffer")

        return buffer

    def get_extra_state(self) -> Any:
        """返回要包含在模块state_dict中的额外状态。
        
        如果你的模块需要存储额外状态，请实现此方法和对应的:func:`set_extra_state`。
        当构建模块的`state_dict()`时会调用此方法。

        注意额外状态必须是可pickle的，以确保state_dict可以正确序列化。
        我们只保证对Tensor的序列化提供向后兼容性；
        其他对象如果其序列化的pickle形式发生变化，可能会破坏向后兼容性。

        返回:
            object: 要存储在模块state_dict中的任何额外状态
        """
        raise RuntimeError(
            "调用了Module.get_extra_state()中不应被调用的代码路径。"
            "请前往 https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "提交issue报告此错误。"
        )

    def set_extra_state(self, state: Any) -> None:
        """设置从加载的`state_dict`中获取的额外状态。
        
        当从:func:`load_state_dict`中加载时，会调用此函数来处理state_dict中的任何额外状态。
        如果你的模块需要在state_dict中存储额外状态，请实现此函数和对应的:func:`get_extra_state`。

        参数:
            state (dict): 来自`state_dict`的额外状态
        """
        raise RuntimeError(
            "调用了Module.set_extra_state()中不应被调用的代码路径。"
            "请前往 https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "提交issue报告此错误。"
        )
    def _apply(self, fn, recurse=True):
        """对模块及其子模块应用函数fn
        
        Args:
            fn: 要应用的函数
            recurse: 是否递归应用到子模块，默认为True
        """
        # 如果recurse为True，递归地对所有子模块应用fn
        if recurse:
            for module in self.children():
                module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            """判断是否应该使用.data赋值方式更新张量
            
            根据新旧张量的类型兼容性和全局设置决定更新方式
            """
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # 如果新张量与现有张量类型兼容
                # 根据全局设置决定是否使用.data赋值方式(未来会改为直接覆盖)
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        # 获取是否应该使用swap_tensors方式的全局设置
        should_use_swap_tensors = (
            torch.__future__.get_swap_module_params_on_conversion()
        )

        # 处理所有参数
        for key, param in self._parameters.items():
            if param is None:
                continue
                
            # 不追踪梯度历史地应用fn到参数
            with torch.no_grad():
                param_applied = fn(param)
                
            # 判断是否应该使用.data赋值方式
            p_should_use_set_data = compute_should_use_set_data(param, param_applied)

            # 判断是否应该使用swap_tensors方式(全局设置或参数是traceable子类)
            p_should_use_swap_tensors = (
                should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)
            )

            param_grad = param.grad
            if p_should_use_swap_tensors:
                # 使用swap_tensors方式交换参数
                try:
                    if param_grad is not None:
                        # 清除梯度以便交换
                        param.grad = None
                    # 创建新参数并交换
                    param_applied = torch.nn.Parameter(
                        param_applied, requires_grad=param.requires_grad
                    )
                    torch.utils.swap_tensors(param, param_applied)
                except Exception as e:
                    if param_grad is not None:
                        param.grad = param_grad
                    raise RuntimeError(
                        f"_apply(): Couldn't swap {self._get_name()}.{key}"
                    ) from e
                out_param = param
            elif p_should_use_set_data:
                # 使用.data赋值方式更新参数
                param.data = param_applied
                out_param = param
            else:
                # 创建新参数对象替换原参数
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            # 处理参数的梯度
            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
                g_should_use_set_data = compute_should_use_set_data(
                    param_grad, grad_applied
                )
                if p_should_use_swap_tensors:
                    # 使用swap_tensors方式交换梯度
                    grad_applied.requires_grad_(param_grad.requires_grad)
                    try:
                        torch.utils.swap_tensors(param_grad, grad_applied)
                    except Exception as e:
                        raise RuntimeError(
                            f"_apply(): Couldn't swap {self._get_name()}.{key}.grad"
                        ) from e
                    out_param.grad = param_grad
                elif g_should_use_set_data:
                    # 使用.data赋值方式更新梯度
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    # 创建新梯度对象
                    assert param_grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(
                        param_grad.requires_grad
                    )

        # 处理所有buffer
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self
    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        """递归地对所有子模块及自身应用函数fn
        
        典型用途包括初始化模型参数(参见:ref:`nn-init-doc`)

        Args:
            fn (Callable[[Module], None]): 要应用于每个子模块的函数

        Returns:
            Module: 返回self以支持链式调用

        示例::
            >>> @torch.no_grad()  # 通常不需要梯度追踪
            >>> def init_weights(m):
            >>>     print(m)  # 打印当前模块
            >>>     if type(m) == nn.Linear:  # 只处理Linear层
            >>>         m.weight.fill_(1.0)  # 权重初始化为1
            >>>         print(m.weight)  # 打印初始化后的权重
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)  # 应用初始化函数
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
            )
        """
        # 递归地对所有子模块应用fn
        for module in self.children():
            module.apply(fn)
        # 最后对自身应用fn
        fn(self)
        return self  # 返回self支持链式调用

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        """将所有模型参数和缓冲区移动到GPU
        
        这会使关联的参数和缓冲区成为不同的对象，因此如果模块将在GPU上优化，
        应在构建优化器之前调用此方法。

        注意:
            此方法会就地(in-place)修改模块。

        Args:
            device (int, optional): 如果指定，所有参数将被复制到该设备

        Returns:
            Module: self
        """
        # 使用_apply方法并传入lambda函数实现CUDA转移
        return self._apply(lambda t: t.cuda(device))
    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
        """将所有模型参数和缓冲区移动到IPU设备
        
        这会使关联的参数和缓冲区成为不同的对象，因此如果模块将在IPU上优化，
        应在构建优化器之前调用此方法。

        注意:
            此方法会就地(in-place)修改模块。

        参数:
            device (int, optional): 如果指定，所有参数将被复制到该设备

        返回:
            Module: 返回self以支持链式调用
        """
        # 使用_apply方法并传入lambda函数实现IPU转移
        return self._apply(lambda t: t.ipu(device))

    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
        """将所有模型参数和缓冲区移动到XPU设备
        
        这会使关联的参数和缓冲区成为不同的对象，因此如果模块将在XPU上优化，
        应在构建优化器之前调用此方法。

        注意:
            此方法会就地(in-place)修改模块。

        参数:
            device (int, optional): 如果指定，所有参数将被复制到该设备

        返回:
            Module: 返回self以支持链式调用
        """
        # 使用_apply方法并传入lambda函数实现XPU转移
        return self._apply(lambda t: t.xpu(device))

    def mtia(self: T, device: Optional[Union[int, device]] = None) -> T:
        """将所有模型参数和缓冲区移动到MTIA设备
        
        这会使关联的参数和缓冲区成为不同的对象，因此如果模块将在MTIA上优化，
        应在构建优化器之前调用此方法。

        注意:
            此方法会就地(in-place)修改模块。

        参数:
            device (int, optional): 如果指定，所有参数将被复制到该设备

        返回:
            Module: 返回self以支持链式调用
        """
        # 使用_apply方法并传入lambda函数实现MTIA转移
        return self._apply(lambda t: t.mtia(device))

    def cpu(self: T) -> T:
        """将所有模型参数和缓冲区移动到CPU
        
        注意:
            此方法会就地(in-place)修改模块。

        返回:
            Module: 返回self以支持链式调用
        """
        # 使用_apply方法并传入lambda函数实现CPU转移
        return self._apply(lambda t: t.cpu())
    
    def type(self: T, dst_type: Union[dtype, str]) -> T:
        """将所有参数和缓冲区转换为指定类型
        
        注意：
            此方法会就地(in-place)修改模块。

        参数：
            dst_type (type或str): 目标数据类型（如torch.float32或'float32'）

        返回：
            Module: 返回self以支持链式调用
        """
        # 使用_apply方法对所有张量应用类型转换
        return self._apply(lambda t: t.type(dst_type))

    def float(self: T) -> T:
        """将所有浮点参数和缓冲区转换为float32类型
        
        注意：
            1. 只影响浮点类型张量
            2. 此方法会就地修改模块

        返回：
            Module: 返回self以支持链式调用
        """
        # 条件转换：仅对浮点类型张量执行float()操作
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        """将所有浮点参数和缓冲区转换为float64类型
        
        注意：
            1. 只影响浮点类型张量
            2. 此方法会就地修改模块

        返回：
            Module: 返回self以支持链式调用
        """
        # 条件转换：仅对浮点类型张量执行double()操作
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self: T) -> T:
        """将所有浮点参数和缓冲区转换为float16类型
        
        注意：
            1. 只影响浮点类型张量
            2. 此方法会就地修改模块
            3. 可能损失数值精度

        返回：
            Module: 返回self以支持链式调用
        """
        # 条件转换：仅对浮点类型张量执行half()操作
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self: T) -> T:
        """将所有浮点参数和缓冲区转换为bfloat16类型
        
        注意：
            1. 只影响浮点类型张量
            2. 此方法会就地修改模块
            3. 专为神经网络优化的16位浮点格式

        返回：
            Module: 返回self以支持链式调用
        """
        # 条件转换：仅对浮点类型张量执行bfloat16()操作
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)
    
    def to_empty(
        self: T, 
        *, 
        device: Optional[DeviceLikeType], 
        recurse: bool = True
    ) -> T:
        """将参数和缓冲区移动到指定设备但不复制存储数据（创建空张量）
        
        该方法会为参数和缓冲区创建新的未初始化张量，保持原张量的形状和类型，
        但存储空间不会被复制（内容将保持未初始化状态）。

        参数：
            device (torch.device): 目标设备（如'cuda:0'）
            recurse (bool): 是否递归处理子模块，默认为True

        返回：
            Module: 返回self以支持链式调用

        典型用例：
            预分配GPU内存但不立即复制数据
        """
        # 使用_apply创建形状相同但未初始化的新张量
        return self._apply(
            lambda t: torch.empty_like(t, device=device),  # 创建同形状的未初始化张量
            recurse=recurse
        )

    @overload
    def to(
        self,
        device: Optional[DeviceLikeType] = ...,
        dtype: Optional[dtype] = ...,
        non_blocking: bool = ...,
    ) -> Self:
        """方法重载声明（类型注解用）
        
        用于类型检查器识别不同的参数组合：
        1. to(device)
        2. to(dtype)
        3. to(device, dtype)
        4. to(tensor)
        """
        ...
    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self:
        """类型转换重载声明（仅dtype参数）"""
        ...

    @overload 
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self:
        """参照张量转换重载声明（根据目标张量属性转换）"""
        ...

    def to(self, *args, **kwargs):
        """移动和/或转换参数与缓冲区的设备/类型
        
        支持四种调用方式：
        1. to(device=None, dtype=None, non_blocking=False)
        2. to(dtype, non_blocking=False) 
        3. to(tensor, non_blocking=False)
        4. to(memory_format=torch.channels_last)

        签名与torch.Tensor.to类似，但仅接受浮点或复数类型。
        整型参数只会移动设备，不会改变类型。

        注意：
            - 此方法会就地修改模块
            - 设置non_blocking可尝试异步转换
            - 从meta设备迁移时需使用to_empty()

        参数：
            device (torch.device): 目标设备
            dtype (torch.dtype): 目标浮点/复数类型
            tensor (Tensor): 参照张量（继承其设备/类型）
            memory_format: 4D张量的内存布局格式

        返回：
            Module: self

        示例：
            >>> # 转换为float64类型
            >>> model.to(torch.double)  
            
            >>> # 移动到GPU并转为半精度
            >>> model.to('cuda', dtype=torch.half)
            
            >>> # 根据参照张量转换
            >>> model.to(example_tensor)
        """
        # 解析输入参数
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        # 类型校验
        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError("仅支持浮点或复数类型")
            if dtype.is_complex:
                warnings.warn(
                    "复数模块是实验性功能，可能存在问题，"
                    "遇到问题请提交issue报告"
                )

        # 定义实际转换函数
        def convert(t):
            try:
                # 处理4D/5D张量的内存格式转换
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                # 常规转换
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None, 
                    non_blocking,
                )
            except NotImplementedError as e:
                if "meta tensor" in str(e):  # 处理meta设备特殊错误
                    raise NotImplementedError(
                        "从meta设备迁移请使用to_empty()方法"
                    ) from None
                raise

        # 应用转换
        return self._apply(convert)
    
    def register_full_backward_pre_hook(
        self,
        hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        """注册模块的反向传播前钩子（full backward pre-hook）
        
        每次计算模块梯度时都会调用该钩子。钩子函数签名应为：
            hook(module, grad_output) -> 新梯度元组或None
        
        注意：
            - grad_output是元组，对应模块输出的梯度
            - 钩子不应修改输入参数，但可以返回替代grad_output的新梯度
            - 非Tensor参数的梯度为None
            - 使用后向钩子时禁止原地修改输入张量

        参数：
            hook: 用户定义的钩子函数
            prepend: 若为True，该钩子将在所有现有钩子之前执行

        返回：
            可移除的句柄，通过handle.remove()删除钩子
        """
        handle = RemovableHandle(self._backward_pre_hooks)  # 创建可移除句柄
        self._backward_pre_hooks[handle.id] = hook  # 存储钩子函数
        if prepend:
            # 将钩子移到字典前端实现优先执行
            self._backward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_backward_hook(
        self, hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:
        """注册旧版反向传播钩子（已弃用）
        
        注意：
            - 未来版本将改变行为
            - 不能与full_backward_hook同时使用
        """
        if self._is_full_backward_hook is True:
            raise RuntimeError("不能同时使用常规和完整反向钩子")
        
        self._is_full_backward_hook = False  # 标记为旧版钩子
        
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_full_backward_hook(
        self,
        hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        """注册完整的反向传播钩子
        
        触发规则：
            1. 通常当计算模块输入梯度时触发
            2. 若输入不需要梯度，则在计算输出梯度时触发
            3. 若输出也不需要梯度，则不触发

        钩子签名：
            hook(module, grad_input, grad_output) -> 新输入梯度或None
        
        注意：
            - grad_input对应位置参数的梯度（忽略kwargs）
            - 禁止原地修改输入/输出张量
            - 不能与旧版backward_hook同时使用

        参数：
            prepend: 若为True，该钩子将优先执行
        """
        if self._is_full_backward_hook is False:
            raise RuntimeError("不能同时使用常规和完整反向钩子")
        
        self._is_full_backward_hook = True  # 标记为完整钩子
        
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        if prepend:
            self._backward_hooks.move_to_end(handle.id, last=False)
        return handle
    def _get_backward_hooks(self):
        """获取当前模块的所有反向传播钩子
        
        返回两个列表：
        1. full_backward_hooks: 完整反向钩子列表
        2. non_full_backward_hooks: 非完整反向钩子列表
        
        注意：
            - 会合并全局钩子和模块本地钩子
            - 根据_is_full_backward_hook标志分类
        """
        full_backward_hooks: list[Callable] = []
        # 收集全局完整钩子
        if _global_is_full_backward_hook is True:
            full_backward_hooks += _global_backward_hooks.values()
        # 收集模块本地完整钩子
        if self._is_full_backward_hook is True:
            full_backward_hooks += self._backward_hooks.values()

        non_full_backward_hooks: list[Callable] = []
        # 收集全局非完整钩子
        if _global_is_full_backward_hook is False:
            non_full_backward_hooks += _global_backward_hooks.values()
        # 收集模块本地非完整钩子
        if self._is_full_backward_hook is False:
            non_full_backward_hooks += self._backward_hooks.values()

        return full_backward_hooks, non_full_backward_hooks

    def _get_backward_pre_hooks(self):
        """获取所有反向传播前钩子
        
        返回合并后的列表（全局+本地）
        """
        backward_pre_hooks: list[Callable] = []
        backward_pre_hooks += _global_backward_pre_hooks.values()
        backward_pre_hooks += self._backward_pre_hooks.values()
        return backward_pre_hooks

    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):
        """检查非完整反向钩子的使用场景并发出警告
        
        参数：
            inputs: 模块的输入张量
            result: 模块的输出张量
            grad_fn: 梯度计算函数
            
        触发警告的场景：
        1. 输入/输出不是张量或张量元组
        2. 输出嵌套在Python数据结构中
        3. 输出由不同计算节点生成
        4. 前向传播包含多个计算节点
        """
        # 检查输出类型
        if not isinstance(result, torch.Tensor):
            if not (
                isinstance(result, tuple)
                and all(isinstance(r, torch.Tensor) for r in result)
            ):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not return a "
                    "single Tensor or a tuple of Tensors is deprecated and will be removed "
                    "in future versions. This hook will be missing some of the grad_output. "
                    "Please use register_full_backward_hook to get the documented behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
                return
        else:
            result = (result,)  # 统一转为元组

        # 检查输入类型
        if not isinstance(inputs, torch.Tensor):
            if not (
                isinstance(inputs, tuple)
                and all(isinstance(i, torch.Tensor) for i in inputs)
            ):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not take as input a "
                    "single Tensor or a tuple of Tensors is deprecated and will be removed "
                    "in future versions. This hook will be missing some of the grad_input. "
                    "Please use register_full_backward_hook to get the documented behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
                return
        else:
            inputs = (inputs,)  # 统一转为元组

        # 检查输出计算节点
        out_grad_fn = {r.grad_fn for r in result if r.grad_fn is not None}
        if len(out_grad_fn) == 0 or (len(out_grad_fn) == 1 and grad_fn not in out_grad_fn):
            warnings.warn(
                "当输出嵌套在Python数据结构中使用非完整反向钩子已被弃用，"
                "该钩子将丢失部分grad_output",
                FutureWarning,
                stacklevel=2,
            )
        elif len(out_grad_fn) > 1:
            warnings.warn(
                "当输出由不同计算节点生成时使用非完整反向钩子已被弃用，"
                "该钩子将丢失部分grad_output。请改用完整反向钩子",
                FutureWarning,
                stacklevel=2,
            )
        else:
            # 检查输入计算节点
            inputs_grad_fn = {i.grad_fn for i in inputs if i.grad_fn is not None}
            next_functions = {n[0] for n in grad_fn.next_functions}
            
            if inputs_grad_fn != next_functions:
                warnings.warn(
                    "当前向传播包含多个计算节点时使用非完整反向钩子已被弃用，"
                    "该钩子将丢失部分grad_input。请改用完整反向钩子",
                    FutureWarning,
                    stacklevel=2,
                )
    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, tuple[Any, ...]], Optional[Any]],
            Callable[
                [T, tuple[Any, ...], dict[str, Any]],
                Optional[tuple[Any, dict[str, Any]]],
            ],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""注册前向传播预处理钩子

        功能特性：
        - 在forward()执行前拦截并可能修改输入参数
        - 支持两种调用签名（是否处理关键字参数）
        - 提供执行顺序控制和安全注销机制

        参数说明：
            hook: 用户定义的钩子函数，有两种形式：
                1. 基础形式: hook(module, args) -> 修改后的输入或None
                2. 增强形式: hook(module, args, kwargs) -> (修改后的输入, kwargs)或None
            prepend: 是否优先执行（插入钩子队列头部）
            with_kwargs: 是否向钩子传递关键字参数

        技术实现：
        1. 使用RemovableHandle实现安全的钩子注销
        2. 通过两个字典分别存储钩子和kwargs标记：
        - _forward_pre_hooks: 存储所有钩子函数
        - _forward_pre_hooks_with_kwargs: 标记需要kwargs的钩子
        3. 使用OrderedDict的move_to_end控制执行顺序

        典型应用场景：
        - 输入数据预处理（如标准化）
        - 动态参数调整（如dropout率）
        - 输入有效性校验
        - 调试信息收集

        示例：
            # 修改输入数据
            def transform_input(module, args):
                return args[0] * 2  # 所有输入放大2倍
                
            # 处理关键字参数    
            def process_kwargs(module, args, kwargs):
                kwargs['mask'] = kwargs['mask'].float()
                return args, kwargs
                
            handle = model.register_forward_pre_hook(transform_input)
            kwarg_handle = model.register_forward_pre_hook(
                process_kwargs, 
                with_kwargs=True
            )
        """
        handle = RemovableHandle(
            self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, tuple[Any, ...], Any], Optional[Any]],  # 基础钩子：模块、输入参数、输出
            Callable[[T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]],  # 增强钩子：带kwargs
        ],
        *,
        prepend: bool = False,     # 是否优先执行
        with_kwargs: bool = False,  # 是否传递kwargs参数
        always_call: bool = False,  # 是否在异常时仍执行
    ) -> RemovableHandle:
        """注册前向传播后处理钩子（forward hook）
        
        该钩子会在每次forward()计算输出后被调用，主要用于：
        - 监控和修改模块输出
        - 特征提取和中间结果捕获
        - 调试和可视化

        参数说明：
            hook: 钩子函数，有两种形式：
                1. 基础形式: hook(module, args, output) -> 修改后的输出或None
                2. 增强形式: hook(module, args, kwargs, output) -> 修改后的输出或None
            prepend: 是否插入钩子队列头部优先执行
            with_kwargs: 是否向钩子传递原始kwargs参数
            always_call: 是否在forward抛出异常时仍强制执行

        技术实现：
        1. 使用三个字典维护钩子状态：
        - _forward_hooks: 存储所有钩子函数
        - _forward_hooks_with_kwargs: 标记需要kwargs的钩子
        - _forward_hooks_always_called: 标记异常时仍执行的钩子
        2. 通过OrderedDict的move_to_end控制执行顺序

        典型应用场景：
            # 特征提取
            def feature_extractor(module, args, output):
                features[module.name] = output.detach()
                
            # 输出归一化    
            def normalize_output(module, args, kwargs, output):
                return F.normalize(output, p=2)
                
            handle = model.register_forward_hook(feature_extractor)
            norm_handle = model.register_forward_hook(
                normalize_output, 
                with_kwargs=True
            )
        """
        # 创建可移除句柄，关联三个状态字典
        handle = RemovableHandle(
            self._forward_hooks,
            extra_dict=[
                self._forward_hooks_with_kwargs,  # kwargs标记字典
                self._forward_hooks_always_called  # 异常处理标记字典
            ],
        )
        
        # 存储钩子函数
        self._forward_hooks[handle.id] = hook
        
        # 设置钩子属性标记
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True

        # 控制执行顺序
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)
        
        return handle

    def _slow_forward(self, *input, **kwargs):
        """慢速前向传播路径（用于JIT追踪和调试）
        
        在模型编译/追踪阶段使用，主要功能：
        1. 处理JIT追踪时的作用域管理
        2. 提供调试接口支持
        3. 保持与普通forward的兼容性

        参数：
            *input: 可变位置参数
            **kwargs: 可变关键字参数

        返回：
            前向传播计算结果
        """
        # 获取当前JIT追踪状态
        tracing_state = torch._C._get_tracing_state()
        
        # 非追踪状态或ScriptMethod直接调用forward
        if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
            return self.forward(*input, **kwargs)
        
        # 检查是否需要记录作用域（用于调试和可视化）
        recording_scopes = torch.jit._trace._trace_module_map is not None
        if recording_scopes:
            # 从模块映射中获取当前模块的名称（用于作用域标记）
            name = torch.jit._trace._trace_module_map[self] if self in torch.jit._trace._trace_module_map else None
            if name:
                tracing_state.push_scope(name)  # 压入作用域栈
            else:
                recording_scopes = False
        
        try:
            # 执行实际前向计算
            result = self.forward(*input, **kwargs)
        finally:
            # 确保作用域栈平衡
            if recording_scopes:
                tracing_state.pop_scope()  # 弹出作用域
        
        return result

    def _wrapped_call_impl(self, *args, **kwargs):
        """调用实现的包装器
        
        提供编译优化路径和常规路径的切换：
        1. 优先使用编译优化后的实现（如TorchScript）
        2. 回退到标准调用实现
        
        参数：
            *args: 可变位置参数
            **kwargs: 可变关键字参数
            
        返回：
            模块调用结果
        """
        if self._compiled_call_impl is not None:
            # 使用编译优化版本（如经过TorchScript编译）
            return self._compiled_call_impl(*args, **kwargs)
        else:
            # 回退到标准调用实现
            return self._call_impl(*args, **kwargs)
    def _call_impl(self, *args, **kwargs):
        """模块调用的核心实现，处理前向传播的所有钩子逻辑
        
        主要功能：
        1. 管理前向/反向传播钩子的执行顺序
        2. 处理异常情况下的钩子调用
        3. 提供JIT编译兼容性支持
        """
        # 决定使用普通forward还是带追踪的slow_forward
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        
        # 快速路径：没有任何钩子时直接调用forward
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks 
                or self._forward_pre_hooks or _global_backward_pre_hooks 
                or _global_backward_hooks or _global_forward_hooks 
                or _global_forward_pre_hooks):
            return forward_call(*args, **kwargs)

        # 初始化结果和已调用钩子记录
        result = None
        called_always_called_hooks = set()

        def inner():
            """实际执行前向传播和钩子的内部函数"""
            nonlocal result, args, kwargs

            # 1. 收集所有反向传播钩子
            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()
            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

            # 2. 执行前向预处理钩子（修改输入）
            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (*_global_forward_pre_hooks.items(), 
                                    *self._forward_pre_hooks.items()):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        # 处理带kwargs的pre-hook
                        args_kwargs_result = hook(self, args, kwargs)
                        if args_kwargs_result is not None:
                            if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError("forward pre-hook必须返回None或(args, kwargs)元组")
                    else:
                        # 处理普通pre-hook
                        args_result = hook(self, args)
                        if args_result is not None:
                            args = args_result if isinstance(args_result, tuple) else (args_result,)

            # 3. 设置反向钩子（如果需要）
            bw_hook = None
            if full_backward_hooks or backward_pre_hooks:
                bw_hook = BackwardHook(self, full_backward_hooks, backward_pre_hooks)
                args = bw_hook.setup_input_hook(args)

            # 4. 执行实际前向计算
            result = forward_call(*args, **kwargs)

            # 5. 执行前向后处理钩子（修改输出）
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (*_global_forward_hooks.items(), 
                                *self._forward_hooks.items()):
                    # 标记已执行的always_call钩子
                    if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                        called_always_called_hooks.add(hook_id)

                    # 根据类型调用不同签名的hook
                    if hook_id in self._forward_hooks_with_kwargs or hook_id in _global_forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            # 6. 设置输出反向钩子
            if bw_hook:
                if not isinstance(result, (torch.Tensor, tuple)):
                    warnings.warn("反向钩子需要输出为Tensor或Tensor元组")
                result = bw_hook.setup_output_hook(result)

            # 7. 处理非完整反向钩子
            if non_full_backward_hooks:
                var = result
                # 找到结果中的第一个Tensor
                while not isinstance(var, torch.Tensor):
                    if isinstance(var, dict):
                        var = next(v for v in var.values() if isinstance(v, torch.Tensor))
                    else:
                        var = var[0]
                # 注册梯度计算钩子
                grad_fn = var.grad_fn
                if grad_fn is not None:
                    for hook in non_full_backward_hooks:
                        grad_fn.register_hook(_WrappedHook(hook, self))
                    self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        # 异常处理逻辑
        if torch.compiler.is_compiling():  # 编译模式下不处理异常
            return inner()
        
        try:
            return inner()
        except Exception:
            # 异常时强制执行always_call标记的钩子
            for hook_id, hook in _global_forward_hooks.items():
                if hook_id in _global_forward_hooks_always_called and hook_id not in called_always_called_hooks:
                    try:
                        hook_result = hook(self, args, result)
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(f"全局always_call钩子异常被忽略: {str(e)}")
            
            for hook_id, hook in self._forward_hooks.items():
                if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)
                        else:
                            hook_result = hook(self, args, result)
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn(f"模块always_call钩子异常被忽略: {str(e)}")
            
            raise  
    # fmt: on
    __call__: Callable[..., Any] = _wrapped_call_impl

    def __getstate__(self):
        # 获取对象状态字典的副本，用于序列化
        state = self.__dict__.copy()
        # 移除不需要序列化的属性"_compiled_call_impl"
        state.pop("_compiled_call_impl", None)
        return state

    def __setstate__(self, state):
        # 将反序列化得到的状态字典更新到当前对象
        self.__dict__.update(state)

        # 以下代码处理旧版本检查点加载时的兼容性问题
        # 为可能缺失的属性设置默认值
        if "_forward_pre_hooks" not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()  # 前向预钩子
        if "_forward_pre_hooks_with_kwargs" not in self.__dict__:
            self._forward_pre_hooks_with_kwargs = OrderedDict()  # 带参数的前向预钩子
        if "_forward_hooks_with_kwargs" not in self.__dict__:
            self._forward_hooks_with_kwargs = OrderedDict()  # 带参数的前向钩子
        if "_forward_hooks_always_called" not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()  # 总是调用的前向钩子
        if "_state_dict_hooks" not in self.__dict__:
            self._state_dict_hooks = OrderedDict()  # 状态字典钩子
        if "_state_dict_pre_hooks" not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()  # 状态字典预钩子
        if "_load_state_dict_pre_hooks" not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()  # 加载状态字典预钩子
        if "_load_state_dict_post_hooks" not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()  # 加载状态字典后钩子
        if "_non_persistent_buffers_set" not in self.__dict__:
            self._non_persistent_buffers_set = set()  # 非持久化缓冲区集合
        if "_is_full_backward_hook" not in self.__dict__:
            self._is_full_backward_hook = None  # 完整反向钩子标志
        if "_backward_pre_hooks" not in self.__dict__:
            self._backward_pre_hooks = OrderedDict()  # 反向预钩子


    # It is crucial that the return type is not annotated as `Any`, otherwise type checking
    # on `torch.nn.Module` and all its subclasses is largely disabled as a result. See:
    # https://github.com/pytorch/pytorch/pull/115074
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        """
        动态属性访问方法，用于获取模块的参数、缓冲区或子模块
        参数:
            name (str): 要访问的属性名称
        返回:
            Union[Tensor, "Module"]: 返回参数(Tensor)、缓冲区(Tensor)或子模块(Module)
        抛出:
            AttributeError: 当属性不存在时抛出异常
        """
        # 检查_parameters字典是否存在，并查找指定名称的参数
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]  # 返回找到的参数Tensor

        # 检查_buffers字典是否存在，并查找指定名称的缓冲区
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]  # 返回找到的缓冲区Tensor

        # 检查_modules字典是否存在，并查找指定名称的子模块
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]  # 返回找到的子模块Module

        # 如果以上都未找到，抛出属性错误异常
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        """属性设置方法（核心方法，处理模块参数的动态管理）
        
        主要功能：
        1. 自动识别并注册Parameter/Module/Buffer等特殊属性
        2. 维护模块内部状态的一致性
        3. 提供全局注册钩子支持
        
        参数：
            name: 属性名
            value: 属性值（可以是Parameter/Module/Tensor等）
        """
        
        def remove_from(*dicts_or_sets):
            """辅助函数：从多个字典/集合中移除指定key"""
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]  # 字典直接删除
                    else:
                        d.discard(name)  # 集合安全移除

        # 获取当前参数字典（可能在__init__之前为None）
        params = self.__dict__.get("_parameters")
        
        # 情况1：值为Parameter类型
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("必须在Module.__init__()之后才能分配参数")
            
            # 清理其他容器中的同名项（保证唯一性）
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)  # 正式注册参数
            
        # 情况2：参数置为None（删除参数）
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"只能分配Parameter或None到参数'{name}'")
            self.register_parameter(name, value)
            
        else:
            # 获取当前子模块字典
            modules = self.__dict__.get("_modules")
            
            # 情况3：值为Module类型
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError("必须在Module.__init__()之后才能分配子模块")
                
                # 清理其他容器中的同名项
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                
                # 执行全局注册钩子（允许修改value）
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                        
                modules[name] = value  # 注册子模块
                
            # 情况4：子模块置为None（删除子模块）
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"只能分配Module或None到子模块'{name}'")
                    
                # 同样执行全局钩子
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                        
                modules[name] = value
                
            else:
                # 获取当前buffer字典
                buffers = self.__dict__.get("_buffers")
                
                # 情况5：值为Buffer/Tensor类型
                if isinstance(value, Buffer) or (buffers is not None and name in buffers):
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(f"只能分配Buffer/Tensor到'{name}'")
                    
                    # 确定buffer是否持久化
                    persistent = (
                        value.persistent if isinstance(value, Buffer) 
                        else name not in self._non_persistent_buffers_set
                    )
                    
                    # === 兼容性处理（历史原因）===
                    # 处理子类可能没有persistent参数的情况
                    if self.register_buffer is torch.nn.Module.register_buffer:
                        # 标准情况直接调用
                        self.register_buffer(name, value, persistent)
                    else:
                        # 检查子类实现是否支持persistent参数
                        sign = inspect.signature(self.register_buffer)
                        if "persistent" in sign.parameters:
                            self.register_buffer(name, value, persistent)
                        else:
                            if not persistent:
                                raise RuntimeError(
                                    "不支持在不含persistent参数的register_buffer中注册非持久buffer"
                                )
                            # 回退到旧版行为（默认持久化）
                            self.register_buffer(name, value)
                    # === 兼容性处理结束 ===
                    
                else:
                    # 情况6：普通属性设置
                    super().__setattr__(name, value)
    def __delattr__(self, name):
        """
        删除对象属性的特殊方法，用于安全删除模块的参数、缓冲区或子模块
        
        参数:
            name: 要删除的属性名称
        
        处理逻辑:
            1. 如果属性是参数(_parameters字典中的键)，则删除该参数
            2. 如果属性是缓冲区(_buffers字典中的键)，则删除该缓冲区
            同时从_non_persistent_buffers_set中移除该名称(如果是非持久化缓冲区)
            3. 如果属性是子模块(_modules字典中的键)，则删除该子模块
            4. 如果都不是以上情况，则调用父类的__delattr__方法
        """
        if name in self._parameters:
            del self._parameters[name]  # 删除参数
        elif name in self._buffers:
            del self._buffers[name]  # 删除缓冲区
            self._non_persistent_buffers_set.discard(name)  # 从非持久化缓冲区集合中移除
        elif name in self._modules:
            del self._modules[name]  # 删除子模块
        else:
            super().__delattr__(name)  # 调用父类方法处理其他属性

    def _register_state_dict_hook(self, hook):
        """
        注册state_dict方法的后处理钩子函数
        
        参数:
            hook: 钩子函数，需符合特定签名:
                hook(module, state_dict, prefix, local_metadata) -> None or state_dict
        
        返回:
            RemovableHandle: 可移除的钩子句柄
        
        抛出:
            RuntimeError: 如果尝试重复注册相同的钩子函数
        
        说明:
            1. 钩子可以原地修改state_dict或返回新的state_dict
            2. 只有根模块返回的新state_dict会被采纳
        """
        if getattr(hook, "_from_public_api", False):
            raise RuntimeError(
                "Cannot register the same function as the state dict post hook that was "
                "previously registered via register_state_dict_post_hook"
            )
        # 创建可移除的钩子句柄并注册
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_post_hook(self, hook):
        r"""Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None

        The registered hooks can modify the ``state_dict`` inplace.
        """
        # In _register_state_dict_hook there was a bug described in
        # https://github.com/pytorch/pytorch/issues/117437 where the return value
        # was only respected for the root module but not child submodules.
        # We fix this in this public version by only allowing inplace modifications on
        # the state_dict by the hook. However, since hooks registered via both these
        # APIs will be added to `_state_dict_hooks` and the type of `_state_dict_hooks`
        # cannot be changed due to many dependencies on it, we mark a hook
        # as being registered via the public API by setting `_from_public_api` on it.
        # In the implementation of `state_dict`, if the callable does not have this
        # flag, the old behavior of respecting the return value will be preserved
        # for the root module, otherwise, we ensure that the hook returns None.
        hook._from_public_api = True
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, prefix, keep_vars) -> None

        The registered hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
        handle = RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "get_extra_state", Module.get_extra_state)
            is not Module.get_extra_state
        ):
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar("T_destination", bound=dict[str, Any])

    @overload
    def state_dict(
        self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...
    ) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> dict[str, Any]:
        ...

    # TODO: Change `*args` to `*` and remove the corresponding warning in docs when BC allows.
    # Also remove the logic for arg parsing together.
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if not getattr(hook, "_from_public_api", False):
                if hook_result is not None:
                    destination = hook_result
            else:
                if hook_result is not None:
                    raise RuntimeError("state_dict post-hook must return None")
        return destination

    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        r"""See :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` for details.

        A subtle difference is that if ``with_module`` is set to ``False``, then the
        hook will not take the ``module`` as the first argument whereas
        :meth:`~torch.nn.Module.register_load_state_dict_pre_hook` always takes the
        ``module`` as the first argument.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(
            hook, self if with_module else None
        )
        return handle

    def register_load_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
        """
        return self._register_load_state_dict_pre_hook(hook, with_module=True)

    def register_load_state_dict_post_hook(self, hook):
        r"""Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

        This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers` that indicates whether keys should be
        assigned their corresponding tensor in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not torch.overrides.is_tensor_like(input_param):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        "expected torch.Tensor or Tensor-like object from checkpoint but "
                        f"received {type(input_param)}"
                    )
                    continue

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = torch.nn.parameter.is_lazy(param)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if (
                    not is_param_lazy
                    and len(param.shape) == 0
                    and len(input_param.shape) == 1
                ):
                    input_param = input_param[0]

                if not is_param_lazy and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        f"size mismatch for {key}: copying a param with shape {input_param.shape} from checkpoint, "
                        f"the shape in current model is {param.shape}."
                    )
                    continue

                if (
                    param.is_meta
                    and not input_param.is_meta
                    and not assign_to_params_buffers
                ):
                    warnings.warn(
                        f"for {key}: copying from a non-meta parameter in the checkpoint to a meta "
                        "parameter in the current model, which is a no-op. (Did you mean to "
                        "pass `assign=True` to assign items in the state dictionary to their "
                        "corresponding key in the module instead of copying them in place?)"
                    )

                try:
                    with torch.no_grad():
                        if use_swap_tensors:
                            new_input_param = param.module_load(
                                input_param, assign=assign_to_params_buffers
                            )
                            if id(new_input_param) == id(input_param) or id(
                                new_input_param
                            ) == id(param):
                                raise RuntimeError(
                                    "module_load returned one of self or other, please .detach() "
                                    "the result if returning one of the inputs in module_load"
                                )
                            if isinstance(param, torch.nn.Parameter):
                                if not isinstance(new_input_param, torch.nn.Parameter):
                                    new_input_param = torch.nn.Parameter(
                                        new_input_param,
                                        requires_grad=param.requires_grad,
                                    )
                                else:
                                    new_input_param.requires_grad_(param.requires_grad)
                            torch.utils.swap_tensors(param, new_input_param)
                            del new_input_param
                        elif assign_to_params_buffers:
                            # Shape checks are already done above
                            if isinstance(param, torch.nn.Parameter):
                                if not isinstance(input_param, torch.nn.Parameter):
                                    input_param = torch.nn.Parameter(
                                        input_param, requires_grad=param.requires_grad
                                    )
                                else:
                                    input_param.requires_grad_(param.requires_grad)
                            setattr(self, name, input_param)
                        else:
                            param.copy_(input_param)
                except Exception as ex:
                    action = "swapping" if use_swap_tensors else "copying"
                    error_msgs.append(
                        f'While {action} the parameter named "{key}", '
                        f"whose dimensions in the model are {param.size()} and "
                        f"whose dimensions in the checkpoint are {input_param.size()}, "
                        f"an exception occurred : {ex.args}."
                    )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "set_extra_state", Module.set_extra_state)
            is not Module.set_extra_state
        ):
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix) :].split(".", 1)
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict` unless
            :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When set to ``False``, the properties of the tensors
                in the current module are preserved whereas setting it to ``True`` preserves
                properties of the Tensors in the state dict. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                for which the value from the module is preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata["assign_to_params_buffers"] = assign
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)  # noqa: F821

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Tensor]]:
        r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for _name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Module"]:
        r"""Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset gradients of all model parameters.

        See similar function under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def share_memory(self: T) -> T:
        r"""See :meth:`torch.Tensor.share_memory_`."""
        return self._apply(lambda t: t.share_memory_())

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Return the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters themselves, the replicas reference the original
        # module.
        replica._parameters = {}
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True  # type: ignore[assignment]

        return replica

    def compile(self, *args, **kwargs):
        """
        Compile this Module's forward using :func:`torch.compile`.

        This Module's `__call__` method is compiled and all arguments are passed as-is
        to :func:`torch.compile`.

        See :func:`torch.compile` for details on the arguments for this function.
        """
        self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)
