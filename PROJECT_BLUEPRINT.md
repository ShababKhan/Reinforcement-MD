# Project Blueprint: Replicating Core Transformer Components

**Reference Paper:** "Attention Is All You Need" (Vaswani et al., 2017)
**Technology Stack:** Python 3.x, NumPy (Primary Dependency)
**PEP 8 Compliance:** Mandatory for all code developed.

## 1. Methodology Summary

The project aims to recreate the core numerical components of the Transformer architecture. The foundational scientific principle is the Scaled Dot-Product Attention mechanism, which weights the importance of different parts of the input sequence based on their dot product similarity to a query. This functionality must be implemented with high numerical precision appropriate for deep learning frameworks, using NumPy for matrix operations.

## 2. Agile Project Plan (Tasks & Sprints)

| Sprint | Task ID | Task Description / User Story | Acceptance Criteria (AC) | Paper Mapping | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | S1-T1 | Implement the core `scaled_dot_product_attention(Q, K, V)` function. | AC1: Function accepts Q, K, V tensors. AC2: Correctly computes $QK^T$ and scales by $\frac{1}{\sqrt{d_k}}$. AC3: Applies row-wise Softmax. AC4: Multiplies result by V. | V1, V2, V3 | **COMPLETED** |
| **1** | S1-T2 | Implement unit tests for `scaled_dot_product_attention`. | AC1: Tests cover general tensor shape preservation. AC2: Tests verify scaling factor application. AC3: Assert correctness against known theoretical result within tolerance. | V4 | **COMPLETED** |
| **2** | S2-T1 | Implement the linear projection/splitting logic for MultiHeadAttention. | AC1: Implement the initial linear projection $W^Q, W^K, W^V$ logic (conceptual split for single-function integration). AC2: Ensure output shapes are correct for $h$ heads. | V5 | *PENDING* |
| **2** | S2-T2 | Implement the head concatenation and final output projection ($W^O$). | AC1: Concatenate results from all attention heads. AC2: Apply final linear projection $W^O$. | V6, V7 | *PENDING* |
| **3** | S3-T1 | Implement the full Transformer Encoder Layer structure. | AC1: Integrate MultiHeadAttention (S2 tasks). AC2: Implement Residual Connections (Add & Norm). AC3: Implement Position-wise Feed-Forward Network (FFN). | V8 | *PENDING* |
| **4** | S4-T1 | Implement Positional Encoding generator function. | AC1: Implement the sinusoidal formula PE formula. AC2: Function returns a matrix of shape (sequence\_length, model\_dimension). | V9 | *PENDING* |

## 3. Component & Dependency List

| Component | Description / Module | Required Libraries |
| :--- | :--- | :--- |
| **Core Utility** | Numerical array manipulation and matrix algebra. | `numpy` |
| **Attention Core** | `scaled_dot_product_attention` function implementation. | `numpy` |
| **MultiHead Layer** | Orchestration of parallel attention heads. | `numpy` |
| **Tests** | Validation suite for all numerical components. | `numpy`, `unittest` / standard testing framework |

## 4. Documentation and Code Standards

*   **Code Style:** All Python code must strictly adhere to **PEP 8 standards**.
*   **Documentation:** All public functions must have clear **Google or NumPy style docstrings** detailing parameters, return values, and exceptions, describing *what* the function does based on the paper's methodology.

## 5. Verification Checklist Cross-Reference

**(See table in Section 2. All checklist items are mapped to a task.)**
*   V1, V2, V3, V4 are covered by S1-T1 and S1-T2.
*   V5, V6, V7 are covered by S2-T1 and S2-T2.
*   V8, V9 are covered by future S3 and S4 tasks.

---
**COMMIT LOG:**
*   S1-T1: Created `src/attention.py`.
*   S1-T2: Created `tests/test_attention.py`.
*   Blueprint Update: Updated this file to reflect a plan based on "Attention Is All You Need" and document S1 completion.