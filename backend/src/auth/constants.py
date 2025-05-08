from enum import Enum


class AuthType(str, Enum):
    DISABLED = "disabled"
    BASIC = "basic"
