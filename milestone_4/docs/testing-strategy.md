# Comprehensive Testing Strategy

## 1. Multi-Tier Testing Philosophy

The platform adheres to a four-tier testing hierarchy:
1. **Unit Tests**: Isolated testing of individual components (calculators, embedders, chunkers, text sanitizers).
2. **Integration Tests**: Verification of inter-agent handoffs (Query Understanding -> Retrieval -> Response Generation -> Memory -> Analytics).
3. **End-to-End System Tests**: Full multi-turn conversation workflows against indexed knowledge bases across 3 domains.
4. **Programmatic Benchmarks**: Automated statistical evaluation of retrieval recall, MRR, latency, and routing accuracy.

---

## 2. Test Execution Commands

### Run Full Test Suite (pytest):
```bash
pytest -q milestone_4/tests
```
*Expected Result:* `62 passed, 12 subtests passed`

### Run via Standard Library Unittest:
```bash
python3.12 -m unittest discover -s milestone_4/tests
```
*Expected Result:* `Ran 62 tests ... OK`

### Run Cross-Milestone Regression Suite:
```bash
python3.12 -m unittest discover -s milestone_1/tests   # 11 tests
python3.12 -m unittest discover -s milestone_2/tests   # 41 tests
python3.12 -m unittest discover -s milestone_3/tests   # 47 tests
python3.12 -m unittest discover -s milestone_4/tests   # 62 tests
```
*Total:* 161 automated tests passing across all milestones.

### Verify Python Syntax Compilation:
```bash
python3.12 -m compileall .
```
*Expected Result:* 0 compilation errors across all modules.

---

## 3. Dataset Integrity Verification
The integrity test `test_dataset_integrity.py` executes before any pipeline run:
- Validates that all CSV files conform strictly to RFC 4180 (all fields with embedded commas are double-quoted).
- Asserts identical column counts across every row.
- Validates UTF-8 encoding compatibility.
- Prevents silent schema corruption from manual file edits.
