# DCADif
DCADif: Decoupled Conditional Adaptive Time-dynamic Fusion Diffusion Inpainting of Traditional Chinese Mural Paintings

## Project Profile

Our work proposes the DCADif framework, an innovative diffusion model that addresses the critical challenge of disentangling structure and style in the inpainting of traditional Chinese murals. This approach establishes a new benchmark for the digital preservation of cultural heritage. Central to our framework is a Decoupled Conditional Encoder that uses parallel pathways—a CLIP encoder for structural line art and a novel SwinStyle Encoder for artistic features—to achieve orthogonal representations. Furthermore, our Time-Adaptive Feature Fusion (TAFF) module dynamically adjusts the influence of these features over the diffusion timestep, mimicking an expert's coarse-to-fine strategy by prioritizing structure before refining style. Validated on our new, large-scale MuralVerse-S dataset, DCADif demonstrates state-of-the-art performance, effectively bridging the gap between structural accuracy and artistic authenticity.

## Method Overview

We introduce the Decoupled Conditional Adaptive Time-dynamic Fusion framework (DCADif), which for the first time realizes a fine-grained decoupling of structure and style for diffusion-based inpainting, providing a new technological paradigm for the high-fidelity digital preservation of cultural heritage. By integrating a novel Decoupled Conditional Encoder with parallel pathways, a dual-stream mechanism is designed to enhance the model's ability to capture orthogonal representations: a pre-trained CLIP for structural line art and a SwinStyle Encoder for multi-scale artistic features. The framework also introduces a Time-Adaptive Feature Fusion (TAFF) module to improve the model's ability to dynamically modulate guidance throughout the denoising process. Additionally, a composite loss function is employed to effectively resolve the trade-off between pixel-level accuracy and perceptual realism. Based on the self-built large-scale dataset, MuralVerse-S, DCADif achieves state-of-the-art performance, significantly outperforming existing methods. This framework not only provides a powerful tool for restoring damaged murals but also offers new insights into achieving both structural accuracy and artistic authenticity in generative restoration.

## Code

We will upload the training scripts, testing code, and the complete
