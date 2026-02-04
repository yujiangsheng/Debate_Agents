#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义异常类
============

定义智能辩论系统中使用的异常类型，便于错误处理和调试。
"""


class DebateSystemError(Exception):
    """辩论系统基础异常类"""
    pass


class ModelLoadError(DebateSystemError):
    """模型加载失败"""
    pass


class AgentError(DebateSystemError):
    """智能体相关错误"""
    pass


class ToolError(DebateSystemError):
    """工具相关错误"""
    pass


class ConfigError(DebateSystemError):
    """配置相关错误"""
    pass
