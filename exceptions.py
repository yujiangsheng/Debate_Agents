#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自定义异常类"""


class DebateSystemError(Exception):
    """辩论系统基础异常类"""
    pass


class ModelLoadError(DebateSystemError):
    """模型加载失败"""
    pass


class AgentError(DebateSystemError):
    """智能体相关错误"""
    pass
