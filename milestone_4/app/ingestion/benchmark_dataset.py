"""Deterministic Benchmark Dataset definition and domain mapping (Phase 7 & Phase 8).

Guarantees:
- Stable, immutable benchmark dataset definition across evaluation runs
- Exact, deterministic domain assignments for every canonical document
- Domain metadata storage and isolation testing support
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

BENCHMARK_DOCUMENTS: List[Dict[str, str]] = [
    {
        "filename": "ml_overview.txt",
        "domain": "machine_learning",
        "type": "txt",
        "description": "Foundational Machine Learning, Supervised vs Unsupervised, Evaluation Metrics"
    },
    {
        "filename": "ml_algorithms.csv",
        "domain": "machine_learning",
        "type": "csv",
        "description": "Tabular catalog of ML algorithms, complexity, and characteristics"
    },
    {
        "filename": "deep_learning_and_transformers.txt",
        "domain": "machine_learning",
        "type": "txt",
        "description": "Deep learning architectures, backpropagation, attention mechanisms, and Transformers"
    },
    {
        "filename": "computer_networks.txt",
        "domain": "computer_networks",
        "type": "txt",
        "description": "OSI 7-layer model, TCP/IP stack, routing protocols, and transport layer mechanics"
    },
    {
        "filename": "network_protocols.csv",
        "domain": "computer_networks",
        "type": "csv",
        "description": "Catalog of network protocols, port mappings, transport layer, and reliability"
    },
    {
        "filename": "cloud_distributed_systems.txt",
        "domain": "computer_networks",
        "type": "txt",
        "description": "Distributed systems, CAP theorem, consensus, microservices, and cloud networking"
    },
    {
        "filename": "cybersecurity_fundamentals.txt",
        "domain": "cybersecurity",
        "type": "txt",
        "description": "CIA triad, defense-in-depth, zero-trust architecture, and identity management"
    },
    {
        "filename": "cryptography_and_zero_trust.csv",
        "domain": "cybersecurity",
        "type": "csv",
        "description": "Tabular cryptographic mechanisms, symmetric/asymmetric algorithms, and zero trust"
    },
    {
        "filename": "incident_response_and_threats.txt",
        "domain": "cybersecurity",
        "type": "txt",
        "description": "Threat modeling, MITRE ATT&CK, NIST incident response lifecycle, and ransomware"
    }
]

BENCHMARK_DOCUMENT_NAMES: List[str] = [d["filename"] for d in BENCHMARK_DOCUMENTS]

BENCHMARK_DOMAIN_MAPPING: Dict[str, str] = {
    d["filename"]: d["domain"] for d in BENCHMARK_DOCUMENTS
}


def get_deterministic_domain(filename: str, content: Optional[str] = None) -> str:
    """
    Deterministically resolves domain using:
    1. Canonical benchmark registry
    2. Filename heuristic
    3. Content density analysis
    """
    clean_fn = Path(filename).name
    if clean_fn in BENCHMARK_DOMAIN_MAPPING:
        return BENCHMARK_DOMAIN_MAPPING[clean_fn]

    lower_fn = clean_fn.lower()
    if any(k in lower_fn for k in ["ml", "machine_learning", "deep_learning", "neural", "transformer", "algorithm"]):
        return "machine_learning"
    elif any(k in lower_fn for k in ["network", "tcp", "udp", "protocol", "osi", "cloud", "distributed"]):
        return "computer_networks"
    elif any(k in lower_fn for k in ["cyber", "security", "crypto", "zero_trust", "incident", "threat", "mitre", "cia"]):
        return "cybersecurity"

    if content:
        lower_content = content.lower()
        ml_score = sum(lower_content.count(w) for w in ["machine learning", "supervised", "unsupervised", "neural network", "model training", "transformer"])
        net_score = sum(lower_content.count(w) for w in ["tcp", "udp", "osi model", "packet", "bandwidth", "ip address", "latency"])
        cyber_score = sum(lower_content.count(w) for w in ["encryption", "confidentiality", "authentication", "zero trust", "threat", "vulnerability"])

        best = max(ml_score, net_score, cyber_score)
        if best >= 3:
            if best == ml_score:
                return "machine_learning"
            elif best == net_score:
                return "computer_networks"
            else:
                return "cybersecurity"

    return "general"


def load_canonical_benchmark_documents(
    pipeline,
    sample_docs_dir: Optional[Path] = None,
    chunk_size: int = 800,
    overlap: int = 150
) -> Dict[str, Any]:
    """
    Ingests strictly the canonical benchmark documents with deterministic domain metadata.
    Avoids arbitrary workspace files polluting benchmark metrics.
    """
    if sample_docs_dir is None:
        sample_docs_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample_docs"

    ingested_counts: Dict[str, int] = {"machine_learning": 0, "computer_networks": 0, "cybersecurity": 0}
    total_chunks = 0

    for item in BENCHMARK_DOCUMENTS:
        fn = item["filename"]
        expected_domain = item["domain"]
        doc_path = sample_docs_dir / fn
        if not doc_path.exists():
            # Check domain subfolder if present
            doc_path = sample_docs_dir / expected_domain / fn

        if doc_path.exists():
            file_bytes = doc_path.read_bytes()
            res = pipeline.ingest(
                file_bytes=file_bytes,
                filename=fn,
                chunk_size=chunk_size,
                overlap=overlap,
                domain_override=expected_domain
            )
            ingested_counts[expected_domain] += 1
            total_chunks += res.get("chunks", 0)

    return {
        "documents_ingested": len(BENCHMARK_DOCUMENTS),
        "domain_counts": ingested_counts,
        "total_chunks": total_chunks
    }
