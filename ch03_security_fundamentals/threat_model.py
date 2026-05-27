import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class STRIDECategory(Enum):
    SPOOFING            = "Spoofing"
    TAMPERING           = "Tampering"
    REPUDIATION         = "Repudiation"
    INFO_DISCLOSURE     = "Information Disclosure"
    DENIAL_OF_SERVICE   = "Denial of Service"
    ELEVATION           = "Elevation of Privelege"


class RiskLevel(Enum):
    CRITICAL = "Critical"
    HIGH     = "High"
    MEDIUM   = "Medium"
    LOW      = "Low"


class MITREAtlasTactic(Enum):
    RECONNAISSANCE       = "AML.TA0002"
    RESOURCE_DEV        = "AML.TA0001"
    ML_ATTACK_STAGING   = "AML.TA0003"
    EXFILTRATION        = "AML.TA0006"
    IMPACT              = "AML.TA0008"


@dataclass
class Threat:
    id: str
    name: str
    stride: STRIDECategory
    atlas_tactic: MITREAtlasTactic
    risk: RiskLevel
    description: str
    affected_components: str
    mitigations: List[str] = field(default_factory=list) # mutable default type
    notes: str = ""


@dataclass
class ThreatModel:
    system_name: str
    version: str
    components: List[str]
    threats: List[Threat] = field(default_factory=list) # mutable default type

    def add_threat(self, threat: Threat) -> None:
        self.threats.append(threat)

    def get_by_risk(self, risk: RiskLevel) -> List[Threat]:
        return [t for t in self.threats if t.risk == risk]
    
    def get_by_stride(self, stride: STRIDECategory) -> list[Threat]:
        return [t for t in self.threats if t.stride == stride]
    
    def summary(self) -> Dict:
        return {
            "system": self.system_name,
            "version": self.version,
            "total_threats": len(self.threats),
            "by_risk": {
                r.value: len(self.get_by_risk(r)) for r in RiskLevel
            },
            "by_stride": {
                s.value: len(self.get_by_stride(s)) for s in STRIDECategory
            }
        }
    

def build_inference_service_threat_model() -> ThreatModel:
    """
    STRIDE threat model for the ch01 EfficientNet-B3 inference service.
    Maps threats to MITRE ATLAS tactics for industry alignment.
    """
    model = ThreatModel(
        system_name="Chest X-Ray Pneumonia Detection Service",
        version="1.0.0",
        components=[
            "FastAPI inference endpoint (/predict)",
            "EfficientNet-B3 model checkpoint",
            "Image preprocessing pipeline",
            "Model loading and serving layer"
        ]
    )

    model.add_threat(Threat(
        id="T-001",
        name="Model Extraction via API Queries",
        stride=STRIDECategory.INFO_DISCLOSURE,
        atlas_tactic=MITREAtlasTactic.EXFILTRATION,
        risk=RiskLevel.HIGH,
        description="Attacker queries the /predict endpoint repeatedly to "
                    "reconstruct model behavior and extract surrogate model.",
        affected_components="FastAPI inference endpoint (/predict)",
        mitigations=[
            "Rate limiting on /predict endpoint",
            "Query anomaly detection",
            "Prediction confidence rounding"
        ]
    ))

    model.add_threat(Threat(
        id="T-002",
        name="Adversarial Input Evasion",
        stride=STRIDECategory.TAMPERING,
        atlas_tactic=MITREAtlasTactic.IMPACT,
        risk=RiskLevel.CRITICAL,
        description="Attacker submits imperceptible X-Ray images "
                    "that cause the model to misclassify PNEUMONIA as NORMAL.",
        affected_components="Image preprocessing pipeline",
        mitigations=[
            "Input preprocessing defenses",
            "Adversarial training",
            "Ensemble detection"
        ]
    ))

    model.add_threat(Threat(
        id="T-003",
        name="Checkpoint Tampering via Pickle Exploit",
        stride=STRIDECategory.TAMPERING,
        atlas_tactic=MITREAtlasTactic.ML_ATTACK_STAGING,
        risk=RiskLevel.CRITICAL,
        description="Attacker replaces best_checkpoint.pt with a trojaned "
                    "version containing malicious pickle payloads that execute "
                    "on torch.load()",
        affected_components="EfficientNet-B3 model checkpoint",
        mitigations=[
            "ModelScan on every checkpoint load",
            "SHA-256 integrity verification",
            "Signed model artifacts"
        ]
    ))

    model.add_threat(Threat(
        id="T-004",
        name="Training Data Poisoning",
        stride=STRIDECategory.TAMPERING,
        atlas_tactic=MITREAtlasTactic.ML_ATTACK_STAGING,
        risk=RiskLevel.HIGH,
        description="Attacker injects poisoned samples into the training "
                    "dataset to introduce backdoor triggers or degrade "
                    "model performance on specific inputs.",
        affected_components="EfficientNet-B3 model checkpoint",
        mitigations=[
            "Dataset provenance verification",
            "Anomaly detection on training data",
            "Clean-label detection"
        ]
    ))

    model.add_threat(Threat(
        id="T-005",
        name="Membership Inference Attack",
        stride=STRIDECategory.INFO_DISCLOSURE,
        atlas_tactic=MITREAtlasTactic.EXFILTRATION,
        risk=RiskLevel.MEDIUM,
        description="Attacker determines whether specific patient X-rays "
                    "were used in training data by analyzing model confidence "
                    "scores, violating patient privacy.",
        affected_components="FastAPI inference endpoint (/predict)",
        mitigations=[
            "Differential privacy during training",
            "Confidence score rounding",
            "Output perturbation"
        ]
    ))

    model.add_threat(Threat(
        id="T-006",
        name="Denial of Service via Large Image Uploads",
        stride=STRIDECategory.DENIAL_OF_SERVICE,
        atlas_tactic=MITREAtlasTactic.IMPACT,
        risk=RiskLevel.MEDIUM,
        description="Attacker floods /predict with oversized images to "
                    "exhaust GPU memory and crash the inference service.",
        affected_components="FastAPI inference endpoint (/predict)",
        mitigations=[
            "File size limits on upload",
            "Request rate limiting",
            "Input validation before preprocessing"
        ]
    ))

    return model


def print_threat_model(model: ThreatModel) -> None:
    """
    Print formatted threat model report to the console.
    """
    summary = model.summary()

    print("\n" + "=" * 60)
    print(f"THREAT MODEL: {model.system_name}")
    print(f"Version: {model.version}")
    print("=" * 60)

    print(f"\nComponents:")
    for component in model.components:
        print(f" - {component}")

    print(f"\nSummary:")
    print(f" Total threats: {summary['total_threats']}")
    print(f" By Risk Level:")
    for risk, count in summary["by_risk"].items():
        if count > 0:
            print(f" {risk:>10}: {count}")

    print(f"\n By STRIDE Category:")
    for stride, count in summary["by_stride"].items():
        if count > 0:
            print(f" {stride:>25}: {count}")

    print(f"\nThreats:")
    print("-" * 60)
    for threat in sorted(model.threats, key=lambda t: t.risk.value):
        print(f"[{threat.id}] {threat.name}")
        print(f" Risk         : {threat.risk.value}")
        print(f" STRIDE       : {threat.stride.value}")
        print(f" ATLAS        : {threat.atlas_tactic.value}")
        print(f" Component    : {threat.affected_components}")
        print(f" Description  : {threat.description}")
        print(f" Mitigations:")
        for m in threat.mitigations:
            print(f" -{m}")


def export_threat_model(model: ThreatModel, output_path: Path) -> None:
    """
    Export threat model to JSON for downstream consumption.
    Useful once the full threat library is built.
    """
    def serialize(obj):
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Type {type(obj)} not serializeable")
    
    data = {
        "system_name": model.system_name,
        "version": model.version,
        "components": model.components,
        "threats": [
            {
                "id": t.id,
                "name": t.name,
                "stride": t.stride.value,
                "atlas_tactic": t.atlas_tactic.value,
                "risk": t.risk.value,
                "description": t.description,
                "affected_component": t.affected_components,
                "mitigations": t.mitigations,
                "notes": t.notes
            }
            for t in model.threats
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=serialize)

    print(f"\nThreat model exported: {output_path}")

if __name__ == "__main__":
    model = build_inference_service_threat_model()
    print_threat_model(model)
    export_path = Path(__file__).parent / "threat_model.json"
    export_threat_model(model, export_path)