# Project Blueprint: Replicating Self-Attention Mechanism from "Attention Is All You Need"

This blueprint details the plan, dependencies, and methodology required to faithfully reimplement the core Transformer attention mechanism, specifically focusing on the Scaled Dot-Product Attention.

## Introduction
This project aims to recreate the core numerical components of the Transformer architecture as described in Vaswani et al. (2017). Adherence to PEP 8 standards for all Python code and comprehensive documentation is mandatory.

## Methodology Summary
The core methodology revolves around implementing the Scaled Dot-Product Attention mechanism, defined by the formula:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
The implementation will be component-based, starting with the most fundamental mathematical operation and building complexity through sprints.

## Agile Project Plan

**Technology Stack:** Python, NumPy (primary numerical library).

### Sprint 1: Core Attention Components (Focus: Scaled Dot-Product Attention)

| Task ID | User Story / Task Description | Acceptance Criteria (AC) | Maps to Paper Feature |
| :--- | :--- | :--- | :--- |
| S1-T1 | Implement the core `scaled_dot_product_attention(Q, K, V)` function. | AC1: Function accepts Q, K, V tensors. AC2: Computes $QK^T$. AC3: Divides by $\sqrt{d_k}$ (where $d_k$ is determined from K's last dimension). AC4: Applies row-wise softmax. AC5: Multiplies by V. AC6: Passes unit tests for shape preservation and numerical stability within expected tolerance. | Section 3.2, Formula (1) |
| S1-T2 | Implement a test suite for `scaled_dot_product_attention`. | AC1: Tests include zero inputs, boundary conditions, and checks against known manual calculations. AC2: Tests must cover the scaling factor application correctly. | Figure 1 (Conceptual flow) |

### Sprint 2: Multi-Head Attention

| Task ID | User Story / Task Description | Acceptance Criteria (AC) | Maps to Paper Feature |
| :--- | :--- | :--- | :--- |
| S2-T1 | Implement the linear projection logic for splitting Q, K, V into multiple heads. | AC1: Function accepts concatenated Q/K/V and `num_heads`. AC2: Correctly shapes the tensors for parallel processing. | Section 3.2, Formula (2) |
| S2-T2 | Implement the full `multi_head_attention` function, chaining S1-T1. | AC1: Applies attention heads independently. AC2: Concatenates the outputs of all heads. AC3: Applies the final linear projection. | Section 3.2, Formula (2) |

## Component & Dependency List

| Component | Description | Dependencies |
| :--- | :--- | :--- |
| **Core Utility** | Numerical processing and matrix algebra. | `numpy` |
| **Attention Layer** | Implements `scaled_dot_product_attention`. | `numpy` |
| **MultiHead Layer** | Orchestrates parallel attention heads. | `numpy` |
| **Softmax Utility** | Stable implementation of softmax activation. | `numpy` |

**Required Technology Stack:** Python 3.x, NumPy

## Tests
All unit tests will reside in the `tests/` directory and will use standard Python libraries for assertion (e.g., `unittest` or equivalent simple assertions).
