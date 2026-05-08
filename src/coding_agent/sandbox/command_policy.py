"""Policy engine for shell, file, and git actions."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from coding_agent.util.errors import ApprovalRequiredError, PolicyDeniedError


class PolicyDecision(str, Enum):
    allow = "allow"
    deny = "deny"
    require_approval = "require_approval"


class RuleSet(BaseModel):
    allow: list[str] = Field(default_factory=list)
    require_approval: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    shell: RuleSet = Field(default_factory=RuleSet)
    file_ops: RuleSet = Field(default_factory=RuleSet)
    git: RuleSet = Field(default_factory=RuleSet)


class CommandPolicyEngine:
    """Evaluates actions against YAML-backed policy rules."""

    def __init__(self, config: dict[str, Any]):
        self.config = PolicyConfig.model_validate(config)

    def evaluate_shell(self, command: str) -> PolicyDecision:
        return self._evaluate(command, self.config.shell)

    def evaluate_file_op(self, action: str) -> PolicyDecision:
        return self._evaluate(action, self.config.file_ops, regex=False)

    def evaluate_git(self, action: str) -> PolicyDecision:
        return self._evaluate(action, self.config.git, regex=False)

    def enforce(self, decision: PolicyDecision, approval: bool = False) -> None:
        if decision == PolicyDecision.deny:
            raise PolicyDeniedError("Action denied by policy")
        if decision == PolicyDecision.require_approval and not approval:
            raise ApprovalRequiredError("Action requires approval")

    @staticmethod
    def _evaluate(value: str, rules: RuleSet, regex: bool = True) -> PolicyDecision:
        if CommandPolicyEngine._matches(value, rules.deny, regex):
            return PolicyDecision.deny
        if CommandPolicyEngine._matches(value, rules.require_approval, regex):
            return PolicyDecision.require_approval
        if rules.allow and CommandPolicyEngine._matches(value, rules.allow, regex):
            return PolicyDecision.allow
        if rules.allow:
            return PolicyDecision.require_approval
        return PolicyDecision.allow

    @staticmethod
    def _matches(value: str, patterns: list[str], regex: bool) -> bool:
        if regex:
            return any(re.match(pattern, value, flags=re.IGNORECASE) for pattern in patterns)
        return value.lower() in {pattern.lower() for pattern in patterns}

