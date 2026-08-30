"""
Aletheia — src/contracts.py

Loads and validates config/kpi_contracts.yaml into typed dataclasses.
The contract is the source of truth for KPI composition, candidate drivers,
source temporal resolution, and optional causal-chain semantics.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import yaml

CausalRole = Literal["direct", "upstream", "context", "response"]
EffectSign = Literal["positive", "negative", "unknown"]

@dataclass(frozen=True)
class Component:
    name: str
    table: str
    column: str

@dataclass(frozen=True)
class RootDriver:
    name: str
    table: str
    column: str
    explains: str
    max_lag: int
    source_cadence_days: int = 1
    causal_role: CausalRole = "direct"
    mediates_through: str | None = None
    expected_effect_sign: EffectSign = "unknown"

@dataclass(frozen=True)
class KpiContract:
    name: str
    formula: str
    components: list[Component] = field(default_factory=list)
    root_drivers: list[RootDriver] = field(default_factory=list)
    def component_names(self) -> set[str]: return {c.name for c in self.components}
    def drivers_for(self, component_name: str) -> list[RootDriver]: return [d for d in self.root_drivers if d.explains == component_name]
    def max_lag_overall(self) -> int: return max((d.max_lag for d in self.root_drivers), default=0)

class ContractValidationError(ValueError):
    pass

def _parse_component(raw: dict) -> Component:
    for required in ("name","table","column"):
        if required not in raw: raise ContractValidationError(f"component missing required field '{required}': {raw}")
    return Component(raw["name"], raw["table"], raw["column"])

def _parse_root_driver(raw: dict) -> RootDriver:
    for required in ("name","table","column","explains","max_lag"):
        if required not in raw: raise ContractValidationError(f"root_driver missing required field '{required}': {raw}")
    max_lag=raw["max_lag"]
    if not isinstance(max_lag,int) or max_lag<0: raise ContractValidationError(f"root_driver '{raw.get('name')}' has invalid max_lag: {max_lag!r}")
    cadence=raw.get("source_cadence_days",1)
    if not isinstance(cadence,int) or cadence<1: raise ContractValidationError(f"root_driver '{raw.get('name')}' has invalid source_cadence_days: {cadence!r}")
    role=str(raw.get("causal_role","direct")).strip().lower()
    if role not in {"direct","upstream","context","response"}: raise ContractValidationError(f"root_driver '{raw.get('name')}' has invalid causal_role: {role!r}")
    effect_sign=str(raw.get("expected_effect_sign","unknown")).strip().lower()
    if effect_sign not in {"positive","negative","unknown"}: raise ContractValidationError(f"root_driver '{raw.get('name')}' has invalid expected_effect_sign: {effect_sign!r}")
    mediator=raw.get("mediates_through")
    mediator=None if mediator in (None,"") else str(mediator).strip()
    return RootDriver(raw["name"],raw["table"],raw["column"],raw["explains"],max_lag,cadence,role,mediator,effect_sign)  # type: ignore[arg-type]

def _validate_contract(contract: KpiContract) -> None:
    names=contract.component_names()
    if not names: raise ContractValidationError(f"KPI '{contract.name}' declares no components")
    seen_c=set()
    for c in contract.components:
        if c.name in seen_c: raise ContractValidationError(f"KPI '{contract.name}': duplicate component '{c.name}'")
        seen_c.add(c.name)
    seen_d=set(); by_name={d.name:d for d in contract.root_drivers}
    for d in contract.root_drivers:
        if d.explains not in names: raise ContractValidationError(f"KPI '{contract.name}': root_driver '{d.name}' explains undeclared component '{d.explains}'")
        key=(d.name,d.explains)
        if key in seen_d: raise ContractValidationError(f"KPI '{contract.name}': duplicate driver/component declaration {key}")
        seen_d.add(key)
        if d.mediates_through:
            mediator=by_name.get(d.mediates_through)
            if mediator is None: raise ContractValidationError(f"KPI '{contract.name}': '{d.name}' mediates_through unknown driver '{d.mediates_through}'")
            if mediator.explains != d.explains: raise ContractValidationError(f"KPI '{contract.name}': '{d.name}' and mediator '{mediator.name}' must explain same component")

def load_contracts(path: str="config/kpi_contracts.yaml") -> dict[str,KpiContract]:
    with open(path,"r",encoding="utf-8") as fh: raw=yaml.safe_load(fh)
    if not isinstance(raw,dict) or not raw: raise ContractValidationError(f"{path} did not parse to a non-empty mapping")
    out={}
    for name,body in raw.items():
        if "formula" not in body: raise ContractValidationError(f"KPI '{name}' missing 'formula'")
        c=KpiContract(name=name,formula=body["formula"],components=[_parse_component(x) for x in body.get("components",[])],root_drivers=[_parse_root_driver(x) for x in body.get("root_drivers",[])])
        _validate_contract(c); out[name]=c
    return out

_CONTRACTS_CACHE: dict[str,KpiContract] | None = None

def get_contract(name: str,path: str="config/kpi_contracts.yaml") -> KpiContract:
    global _CONTRACTS_CACHE
    if _CONTRACTS_CACHE is None: _CONTRACTS_CACHE=load_contracts(path)
    if name not in _CONTRACTS_CACHE: raise KeyError(f"No KPI contract named '{name}'. Known KPIs: {sorted(_CONTRACTS_CACHE)}")
    return _CONTRACTS_CACHE[name]

def detect_formula_operator(formula: str) -> Literal["multiply","divide","single"]:
    if "=" not in formula: return "single"
    rhs=formula.split("=",1)[1]; mul=("*" in rhs or "×" in rhs); div=("/" in rhs or "÷" in rhs)
    if mul and div: raise ContractValidationError(f"formula mixes multiply and divide: {formula!r}")
    if mul: return "multiply"
    if div: return "divide"
    return "single"
