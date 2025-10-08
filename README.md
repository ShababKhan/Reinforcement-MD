# Reinforced Molecular Dynamics (rMD) Project

This repository contains the code implementation for the Reinforced Molecular Dynamics (rMD) method as described in Kolossváry & Coffey (2025), focusing on modeling the CRBN conformational transition.

## Project Status
This project is initialized and currently executing **Sprint 1**: Data Ingestion and Base Autoencoder (Loss2 only).

**Documentation Status:**
*   `docs/rMD_METHODOLOGY.md`: Created, detailing the dual-loss structure and path generation plan.

**Artifacts Created:**
*   `requirements.txt`: Core dependencies listed.
*   `src/__init__.py`: Initialized package structure.

**Immediate Next Tasks (Sprint 1):**
1.  Implement `data_loader.py` to handle structure loading, heavy-atom extraction, and alignment (M12, M4, M5).
2.  Implement the base Encoder/Decoder network in `model.py` using Swish activation (M1, M2, M3).