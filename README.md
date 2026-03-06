# ECG-Net

**End-to-end ECG image digitisation pipeline for reconstructing clean waveforms and converting them into time-series data.**

ECG-Net frames ECG image digitization as a **structured dense prediction problem** rather than a direct image-to-sequence regression problem.

Instead of asking a network to output a raw waveform vector from an ECG image, ECG-Net first learns a **pixelwise probabilistic representation** of the trace. For each cropped ECG lead panel, the model predicts a **1-channel heatmap** whose high-response pixels correspond to the likely location of the waveform. This predicted heatmap is then converted into a 1D waveform through a decoding step, while the model itself is also trained with differentiable signal-domain supervision.

This is the core ML idea of the project:

* treat the ECG trace as a visual object that is easier to localize spatially than regress numerically from scratch,
* enforce that the localized trace must also decode to a numerically meaningful physiological waveform,
* combine segmentation-style supervision and waveform-style supervision into one training objective.

In that sense, ECG-Net is not just a segmentation model and not just a sequence regressor. It is a **hybrid vision-to-signal learning system**.

At the model level, ECG-Net is built around:

* a **U-Net-like encoder-decoder architecture**,
* a `timm` feature extractor backbone (`tf_efficientnet_b0`),
* soft Gaussian heatmap targets,
* differentiable waveform extraction using softargmax,
* multi-term loss design coupling image-space and signal-space constraints,
* phase-wise training with an additional shift-tolerant alignment loss introduced later in optimization.

---

## 1. Machine Learning Problem Statement

### Task

Given a cropped ECG lead panel image, predict the underlying waveform.

However, ECG-Net does not predict the waveform directly as a vector at first. Instead, it learns the intermediate mapping:

```text
Input image panel
-> trace heatmap
-> differentiable waveform estimate
-> final digitized ECG signal
```

So the primary supervised output of the neural network is:

* a dense heatmap of shape `1 x H x W`

where:

* `H = 256`
* `W = 512`

This heatmap is interpreted column-wise: for each horizontal position, the network expresses where the waveform likely lies vertically.

### Why this formulation is better than direct regression

Directly regressing ECG amplitudes from images is difficult because:

* the waveform is visually thin,
* the background contains grid lines and paper artifacts,
* localization error is spatial before it is numeric,
* the waveform has geometric continuity that direct vector regression may not exploit well.

By predicting a heatmap first, the model is encouraged to solve the easier and more natural subproblem:

> **Where is the trace?**

Then, from that trace representation, the project reconstructs:

> **What is the signal value?**

This decomposition gives the model a stronger inductive bias.

---

## 2. Learning Objective at a High Level

ECG-Net is trained with a **joint objective** that operates in two coupled spaces:

### A. Image space

The predicted heatmap must overlap the target trace heatmap.

This encourages the model to localize the ECG ink path correctly.

### B. Signal space

The same predicted logits must decode into a waveform that matches the ground-truth ECG series in millivolts.

This ensures that the visual prediction is not merely plausible but also physiologically faithful after digitization.

This two-level supervision is the defining ML characteristic of ECG-Net.

A model trained only with segmentation loss could still produce a trace that looks reasonable but shifts slightly, thickens, fragments, or jitters in a way that degrades recovered signal quality. ECG-Net addresses that by explicitly optimizing the waveform derived from the heatmap.

---

## 3. Supervision Unit and Dataset Construction

### Sample granularity

The training sample is **one lead panel**, not one full ECG page.

Each ECG record contributes multiple learning samples corresponding to the standard leads. This is a sensible ML design because the target structure is local and lead-specific.

### Record-level splitting

The notebook performs validation splitting at the **record level**, not panel level.

This matters because multiple lead panels from one ECG sheet are strongly correlated. If panels from the same record were split across train and validation, the validation score would be inflated due to leakage. Splitting by record preserves the integrity of evaluation.

Configured validation fraction:

* `VALID_FRAC = 0.10`

### Returned tensors per sample

Each sample provides:

* input image tensor `x`
* target heatmap `y`
* target waveform `s_mv`
* pixels-per-mV scaling value `ppmv`

This is important because the model is supervised with more than just segmentation masks. The dataset explicitly carries the ingredients required for signal-domain supervision too.

---

## 4. Input Representation

### Panel resolution

Each lead panel is resized to:

* width: `512`
* height: `256`

This fixes a common spatial support for the network and for heatmap generation.

### Base image channels

The standard image input is RGB, scaled to `[0, 1]`.

So the default tensor without auxiliary engineering would be:

* `3 x 256 x 512`

### Auxiliary trace-enhancement channel

ECG-Net optionally adds a fourth channel through:

* `USE_TRACE_CHANNEL = True`

This extra channel is not learned separately; it is handcrafted and then concatenated to the RGB channels before feeding the image into the encoder.

It is built from:

* grayscale conversion,
* black-hat morphology,
* Canny edge extraction,
* Gaussian smoothing,
* weighted fusion.

The fusion weights in the notebook are:

* black-hat weight: `0.70`
* edge weight: `0.30`

Kernel scale is derived from image size via:

* `TRACE_BH_FRAC = 1 / 48`

### ML interpretation of the extra channel

This extra channel acts as **domain-informed feature engineering**.

Since the model uses:

* `PRETRAINED_ENCODER = False`

The encoder does not start with generic visual priors from ImageNet. In that setting, a handcrafted trace-enhancement channel is valuable because it injects low-level bias toward the relevant foreground structure.

In other words, rather than forcing the first convolutional layers to rediscover morphology-like and edge-like filters entirely from data, ECG-Net gives them a stronger starting signal.

That is especially useful for ECG images, where the waveform can be subtle relative to background grid texture.

### Effective model input

With the trace channel enabled, the model receives:

* `4 x 256 x 512`

This is one of the notebook’s strongest domain-specific choices.

---

## 5. Target Representation

### Ground-truth waveform target

For each lead panel, the source ECG series is converted to a fixed-length waveform target of width `512`.

That means the target signal is aligned to the image width dimension.

The waveform target lives in **millivolts**, not arbitrary normalized coordinates. That makes the downstream signal supervision physically meaningful.

### Heatmap target

The waveform is then projected into image space to generate a 2D supervision target.

For each `x`-position:

* the waveform amplitude determines the corresponding `y`-location,
* a Gaussian band is drawn around that `y`-position.

The result is a soft heatmap with shape:

* `256 x 512`

and values in `[0, 1]`.

Configured target blur:

* `GAUSS_SIGMA_PX = 2.0`

### Why soft Gaussian heatmaps are superior to hard lines

A hard one-pixel-wide binary trace would make optimization brittle because:

* ECG traces can be visually thicker than one pixel,
* slight projection mismatch would create large loss jumps,
* class imbalance would become even harsher,
* gradients would be sparse and unstable.

The Gaussian target solves that by creating a smooth probability band around the ideal trace location. This provides:

* denser gradients,
* alignment tolerance,
* better optimization stability,
* more realistic supervision for ambiguous stroke thickness.

From an ML standpoint, this is a very good target design choice.

---

## 6. Model Architecture

### Topology

ECG-Net uses a custom U-Net-style heatmap predictor implemented as:

* `TimmUNetHeatmap`

The structure is:

```text
input
-> encoder
-> bottleneck
-> decoder with skip connections
-> 1-channel heatmap head
```

### Encoder

Backbone:

* `tf_efficientnet_b0`

Construction:

* `timm.create_model(..., features_only=True)`

Settings used:

* `in_chans = 4` when trace channel is enabled
* `out_indices = (1, 2, 3, 4)`
* `pretrained = False`

### Meaning of `features_only=True`

This makes the EfficientNet backbone act as a hierarchical feature pyramid extractor rather than a classifier. Intermediate feature maps from multiple stages are returned and later fused in the decoder.

That is exactly the right mode for segmentation-like tasks.

### Feature hierarchy

The encoder produces four progressively deeper feature maps:

* shallow spatial features,
* intermediate structural features,
* deeper semantic representations,
* deepest compressed context features.

These are then used in skip-connected decoding.

### Bottleneck

The deepest encoder map is passed through two `Conv-BN-SiLU` blocks and projected to `256` channels.

This bottleneck has two purposes:

* refine high-level semantic features,
* standardize channel width before decoder fusion.

### Decoder

The decoder has three stages:

* bilinear upsampling,
* concatenation with the corresponding skip feature map,
* two `Conv-BN-SiLU` blocks.

Channel flow in the notebook is:

* bottleneck output: `256`
* decoder stage 1 output: `128`
* decoder stage 2 output: `64`
* decoder stage 3 output: `32`

This is a lightweight decoder compared with heavier segmentation networks, but it is computationally efficient and probably adequate for thin-curve localization.

### Output head

The decoder ends in a `1x1` convolution that maps the final feature map to:

* one heatmap channel

Then the heatmap is interpolated back to the input spatial size.

### Activation handling

The network outputs raw logits.

This is correct and important:

* `BCEWithLogitsLoss` expects raw logits,
* sigmoid can then be applied only where needed,
* logits remain numerically stable during optimization.

---

## 7. Forward Semantics of the Network

The network is not just classifying pixels independently. Because of the encoder-decoder topology, it learns a hierarchy of representations.

### Early layers

Capture local texture, edges, trace thickness, line contrast, and grid interaction.

### Middle layers

Capture larger waveform motifs, local curvature, stroke continuity, and separation between signal trace and background patterning.

### Deep layers

Encode panel-level context and broader lead structure.

### Decoder

Reprojects that hierarchical information back into spatial coordinates to produce a dense trace likelihood map.

For ECG digitization, this is a strong design because the problem requires both:

* very fine localization precision,
* broader context for continuity and ambiguity resolution.

A purely shallow model would fail on context. A purely deep classifier-style model would lose spatial precision. U-Net-style fusion is a natural fit.

---

## 8. Differentiable Waveform Extraction

This is one of the most important ML mechanisms in ECG-Net.

### Objective

Convert the predicted 2D logits into a differentiable 1D waveform estimate so that waveform-domain losses can backpropagate into the segmentation network.

### Method: softargmax over height

Given logits of shape:

* `B x 1 x H x W`

The notebook applies a softmax along the height dimension for each column.

So for each `x`-position, the model produces a probability distribution over all possible `y`-locations of the trace.

Then it computes the expected row index:

```text
y_soft = sum(p(y) * y)
```

This is effectively a softargmax, because it approximates choosing the most likely row while remaining differentiable.

### Temperature

The sharpness of this distribution is controlled by:

* `SOFTARGMAX_TEMP = 10.0`

A higher temperature makes the distribution more concentrated near the strongest logit response, approximating hard argmax more closely without breaking differentiability.

### Conversion to millivolts

Once the expected `y`-coordinate is computed, the notebook converts it into waveform amplitude using:

* panel midpoint,
* per-sample pixels-per-mV calibration.

This yields a differentiable predicted waveform:

* `s_pred_mv`

### Why softargmax is important

Without this step, the model could only be supervised through heatmap overlap. Softargmax creates a direct bridge:

```text
heatmap logits
-> differentiable waveform
-> signal-domain loss
-> gradient back into the heatmap predictor
```

This is arguably the most elegant ML choice in the notebook.

It lets ECG-Net optimize for the real downstream objective while still training through a dense visual intermediate representation.

---

## 9. Loss Function Design

ECG-Net uses a composite objective combining multiple terms that each address a different failure mode.

### 9.1 Heatmap reconstruction loss

#### BCEWithLogitsLoss

The primary segmentation term is binary cross-entropy on the heatmap logits.

Positive class weighting is used:

* `POS_WEIGHT = 6.0`

This matters because the ECG trace occupies only a tiny fraction of pixels. Without class weighting, the model could minimize loss by favoring background almost everywhere.

#### Dice loss

The notebook adds a Dice-style overlap loss:

* `0.7 x dice_loss_with_logits`

Dice complements BCE because:

* BCE is pointwise,
* Dice is overlap-aware,
* Dice helps reduce fragmentation and underprediction of thin structures.

#### Combined image-space loss

The effective heatmap loss is:

```text
loss_heat = BCEWithLogits + 0.7 * Dice
```

This is a good pairing for thin foreground segmentation.

### 9.2 Waveform L1 loss

The predicted waveform from softargmax is directly compared to the ground-truth waveform in millivolts.

Base form:

* L1 error between predicted and target waveform

This term encourages numeric accuracy after decoding, not just visual alignment.

#### Energy-aware weighting

The notebook scales waveform L1 by a signal-energy factor:

```text
energy = mean(y^2)
weight = energy / (energy + ENERGY_K)
```

with:

* `ENERGY_K = 2e-4`

This is a subtle but smart design choice.

Why it helps:

* very low-energy signals may be dominated by near-flat baseline regions,
* their raw L1 error can be noisy or less informative,
* energy weighting reduces overemphasis on samples where waveform shape is weak or trivial.

The waveform term is then weighted globally by:

* `WAVE_L1_W = 0.22`

### 9.3 Smoothness priors

The predicted waveform is regularized using finite-difference penalties.

#### First derivative prior

The notebook computes:

```text
d1 = s[t] - s[t-1]
```

and penalizes its absolute magnitude.

Weight:

* `SMOOTH_D1_W = 0.02`

#### Second derivative prior

It also computes:

```text
d2 = s[t] - 2*s[t-1] + s[t-2]
```

and penalizes that too.

Weight:

* `SMOOTH_D2_W = 0.01`

#### Why these priors matter

Heatmap predictions can look acceptable while still producing jittery columnwise waveform extraction. The smoothness losses suppress:

* spiky per-column noise,
* unnatural oscillation,
* staircase artifacts,
* local instability in decoded amplitude.

These priors are especially helpful because the waveform is derived from a per-column expectation, which can be sensitive to heatmap noise.

So these terms stabilize the learned signal without requiring a separate recurrent or sequence model.

### 9.4 Shift-tolerant alignment loss

This is another sophisticated design element.

#### Motivation

Even when the predicted waveform shape is correct, small horizontal offsets may appear because of:

* slight crop mismatch,
* trace thickness ambiguity,
* image-wave alignment noise,
* local visual uncertainty.

A strict pointwise L1 loss penalizes such near-correct predictions too harshly.

#### Mechanism

The notebook defines a loss that compares the predicted and ground-truth waveforms across multiple small horizontal shifts.

For each shift `s` in a bounded range:

* align prediction and ground truth with that shift,
* compute mean absolute error,
* apply a baseline correction term,
* collect the error for that alignment.

Then instead of taking a hard minimum, the notebook uses a soft minimum via log-sum-exp:

```text
softmin(E) = -tau * logsumexp(-E / tau)
```

with:

* `SHIFT_TAU = 0.035`

This makes the operation smooth and differentiable.

#### Maximum shift

The allowed shift window is controlled by:

* `SHIFT_MAX_FRAC = 0.010`

For width `512`, that is a small alignment tolerance.

#### Global weight

The shift-tolerant loss enters the training objective with:

* `SHIFT_LOSS_W = 0.55`

#### Why this is strong

This makes ECG-Net less brittle to tiny phase mismatches while still requiring morphological correctness.

It is especially suitable for digitization tasks where the exact spatial alignment between image trace and waveform samples may not be perfect.

---

## 10. Curriculum / Two-Phase Training Strategy

The notebook does not activate all losses equally from the first epoch.

### Phase 1

For the first:

* `PHASE1_EPOCHS = 10`

The network learns mainly from:

* heatmap reconstruction,
* waveform L1,
* smoothness priors.

This lets the model first learn the core task:

* detect the trace,
* decode a basic waveform,
* avoid severe instability.

### Phase 2

After epoch 10:

* learning rate is reduced,
* shift-tolerant alignment loss is activated.

Learning-rate multiplier:

* `PHASE2_LR_MULT = 0.35`

This means training becomes more conservative just as the objective becomes more refined.

### Why this curriculum makes sense

If shift-tolerant alignment loss were activated too early, the model might exploit alignment flexibility before learning good localization. By delaying it, ECG-Net first learns a stable coarse solution, then later optimizes robustness and alignment quality.

This is a good curriculum-style optimization decision.

---

## 11. Total Training Objective

The full loss changes by phase.

### Phase 1 loss

```text
loss = heatmap_loss + WAVE_L1_W * wave_l1 + SMOOTH_D1_W * d1 + SMOOTH_D2_W * d2
```

### Phase 2 loss

```text
loss = heatmap_loss + WAVE_L1_W * wave_l1 + SHIFT_LOSS_W * shift_term + SMOOTH_D1_W * d1 + SMOOTH_D2_W * d2
```

### Interpretation of the total objective

Each component has a clear role:

* **BCE**: pixelwise foreground/background discrimination
* **Dice**: structural overlap of the thin trace
* **Wave L1**: amplitude fidelity in physiological units
* **D1**: suppress local jitter
* **D2**: suppress high-curvature noise
* **Shift-tolerant term**: permit tiny alignment ambiguity while preserving waveform shape

This is not a generic off-the-shelf segmentation loss. It is clearly tailored to ECG digitization as a vision-to-signal reconstruction problem.

---

## 12. Optimization Strategy

### Optimizer

The notebook uses:

* `AdamW`

with:

* learning rate: `3e-4`
* weight decay: `1e-2`

This is a strong default for modern vision training.

### Batch size

* `BATCH = 12`

### Gradient clipping

Maximum norm:

* `GRAD_CLIP = 5.0`

This is helpful because the objective combines multiple loss terms, and softargmax-based signal supervision can occasionally produce unstable gradients. Clipping keeps optimization controlled.

### Mixed precision

AMP is enabled on CUDA:

* `AMP = True` when CUDA is available

This improves speed and memory efficiency while preserving the training structure.

### TF32 behavior

The notebook also enables TF32 acceleration where available on CUDA backends. That is an implementation detail, but it supports efficient training.

### Manual LR schedule

There is no cosine scheduler or one-cycle schedule. Instead, the notebook uses a simple phase transition:

* keep original LR through phase 1,
* multiply by `0.35` at phase 2.

This is simple, interpretable, and coordinated with the loss curriculum.

---

## 13. Data Augmentation Strategy

The augmentation policy is intentionally light.

During training:

* with probability `0.50`, random brightness/contrast perturbation is applied,
* with probability `0.15`, Gaussian blur is applied.

### What these augmentations simulate

They mostly target:

* contrast variability,
* scanner brightness differences,
* print darkness variation,
* slight image softness.

### What is not present

The notebook does not use aggressive geometric augmentation such as:

* random rotations,
* perspective transforms,
* elastic warping,
* severe noise injection,
* stain overlays,
* handwriting simulation.

### ML implication

This suggests the project relies more on:

* deterministic preprocessing,
* panel rectification,
* consistent layout assumptions,

and less on augmentation-driven invariance.

That is workable, but it likely limits robustness under heavy real-world variation.

---

## 14. Validation Design

Validation in ECG-Net is not limited to a loss scalar. It includes multiple complementary metrics.

### 14.1 Validation loss

The notebook reports validation loss using:

* heatmap loss,
* waveform L1,
* smoothness priors.

It does not include the shift-tolerant loss in the reported validation loss scalar in the same weighted way it appears during late training.

So validation loss is informative, but not the ultimate model-selection criterion.

### 14.2 Trace localization accuracy

The notebook computes:

* `trace_acc@2px`

For each image column:

* predicted row is derived from argmax of predicted heatmap,
* target row is derived from argmax of target heatmap,
* accuracy is the fraction of columns whose absolute row error is within 2 pixels.

This is a geometric localization metric.

It measures whether the predicted trace is spatially close to the target trace.

### 14.3 Signal reconstruction metric

The most important validation metric is:

* **best-shift SNR (dB)**

The decoded predicted waveform is compared to the ground-truth waveform after allowing a small shift.

This is critical because it reflects actual digitization quality more directly than segmentation overlap alone.

Low-energy targets are skipped using:

* `SNR_SKIP_ENERGY_THR = 1e-5`

That prevents unstable or uninformative SNR estimates for nearly flat signals.

### 14.4 Checkpoint selection

The best model is selected by:

* highest validation SNR

not lowest validation loss.

This is exactly the right choice for this task because the real objective is signal fidelity, not just mask accuracy.

A model with slightly worse heatmap overlap but better recovered waveform should be preferred, and the notebook’s selection strategy captures that.

---

## 15. Inference Philosophy from the ML Perspective

ECG-Net is not a purely end-to-end waveform regressor.

At inference time, the model predicts a heatmap, but final waveform extraction is handled by a deterministic decoder rather than by another neural network layer.

That means the full system is:

* partially learned,
* partially algorithmic.

### Why this hybrid design is reasonable

For this problem, deterministic path extraction has real advantages:

* it enforces continuity,
* it stabilizes trace selection,
* it reduces dependence on fully learned sequence decoding,
* it exploits geometric structure explicitly.

From an ML systems viewpoint, ECG-Net combines:

* learned perceptual localization
* structured non-learned decoding

That is often stronger than insisting on fully neural end-to-end regression when the geometry is well understood.

---

## 16. Key Hyperparameters

### Model and input

* Encoder: `tf_efficientnet_b0`
* Pretrained encoder: `False`
* Input channels: `4` with trace channel
* Output channels: `1`
* Panel width: `512`
* Panel height: `256`

### Heatmap target

* Gaussian sigma: `2.0`
* Positive class weight: `6.0`

### Waveform supervision

* Softargmax temperature: `10.0`
* Wave L1 weight: `0.22`
* Smooth D1 weight: `0.02`
* Smooth D2 weight: `0.01`
* Energy constant: `2e-4`

### Shift-tolerant alignment

* Enabled: `True`
* Shift loss weight: `0.55`
* Tau: `0.035`
* Max shift fraction: `0.010`

### Training

* Epochs: `25`
* Batch size: `12`
* Learning rate: `3e-4`
* Weight decay: `1e-2`
* Gradient clip: `5.0`
* Validation fraction: `0.10`
* Phase 1 epochs: `10`
* Phase 2 LR multiplier: `0.35`

### Auxiliary channel

* Trace channel enabled: `True`
* Black-hat contribution: `0.70`
* Edge contribution: `0.30`

---

## 17. Strengths of ECG-Net from an ML Research Perspective

### 17.1 Strong problem decomposition

The model solves a visually natural intermediate task first: localize the trace. This is better aligned with the image domain than direct waveform regression.

### 17.2 Multi-space supervision

By supervising both the heatmap and the decoded waveform, ECG-Net avoids the common pitfall of visually plausible but numerically poor outputs.

### 17.3 Differentiable visual-to-signal bridge

Softargmax is an elegant mechanism that connects segmentation and regression without breaking end-to-end differentiation through the learnable part.

### 17.4 Domain-informed inductive bias

The handcrafted trace channel is a good example of useful prior knowledge aiding a vision model.

### 17.5 Well-designed thin-structure loss

Combining BCE and Dice is appropriate for sparse trace localization.

### 17.6 Late-stage alignment robustness

Shift-tolerant supervision improves tolerance to slight crop/phase mismatch without requiring full DTW-style complexity.

### 17.7 Evaluation aligned with actual task

Checkpointing by SNR rather than raw loss is a very strong practical choice.

---

## 18. Limitations of the Current ML Design

### 18.1 No pretrained backbone

Using `pretrained=False` may limit representation quality, especially if the training set is not large enough to learn robust low-level and mid-level features from scratch.

### 18.2 Limited augmentation

The model may not generalize well to:

* severe rotation,
* nonstandard print artifacts,
* handwriting,
* heavy blur,
* stamps,
* damaged paper,
* varied scanner noise.

### 18.3 Single split, no cross-validation

A one-shot random validation split may produce optimistic or pessimistic estimates depending on sample composition.

### 18.4 Lightweight decoder

The decoder is efficient but not especially expressive. More complex cluttered cases might benefit from richer feature fusion or attention.

### 18.5 Hybrid, not fully learned decoding

The final waveform extraction is not fully differentiable end-to-end across the whole inference path. The neural model is optimized against softargmax waveform loss, but final inference still relies on a separate deterministic path extractor.

### 18.6 No explicit uncertainty modeling

The network predicts a heatmap, but there is no dedicated uncertainty estimation or confidence calibration for downstream trustworthiness.
