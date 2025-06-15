### Remote Sensing Captioning with PaliGemma and QLoRA: A Progressive Fine-Tuning Strategy

This repository contains the implementation and experimental pipeline for fine-tuning [PaliGemma-3B](https://huggingface.co/google/paligemma-3b) on the RSICD dataset using Low-Rank Adaptation (LoRA) and 4-bit quantization (QLoRA). The goal is to generate high-quality captions for remote sensing imagery with minimal compute.

The project introduces a **three-tier progressive grid search** strategy that:

* Identifies the most effective architecture adaptation mode (encoder, decoder, or full)
* Optimizes LoRA hyperparameters (rank, scaling factor, target modules)
* Tunes training dynamics under compute constraints

The best model—using **decoder-only LoRA**, **full adapter injection**, and just **5 epochs** of training—achieved:

* **BLEU**: 0.3420
* **ROUGE-L**: 0.4842
* **METEOR**: 0.4802

All while fine-tuning fewer than 0.05% of total parameters.
