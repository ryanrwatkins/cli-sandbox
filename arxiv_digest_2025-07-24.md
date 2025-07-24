# arXiv Digest for 2025-07-24

Top 3 categories for you: ['cs.LO', 'cs.CV', 'cs.FW']

## Transformer Based Building Boundary Reconstruction using Attraction Field Maps

**Link:** [https://arxiv.org/pdf/2507.17038](https://arxiv.org/pdf/2507.17038)

**Similarity:** 0.3395

### Abstract

In recent years, the number of remote satellites orbiting the Earth has grown significantly, streaming vast amounts of high-resolution visual data to support diverse applications across civil, public, and military domains. Among these applications, the generation and updating of spatial maps of the built environment have become critical due to the extensive coverage and detailed imagery provided by satellites. However, reconstructing spatial maps from satellite imagery is a complex computer vision task, requiring the creation of high-level object representations, such as primitives, to accurately capture the built environment. While the past decade has witnessed remarkable advancements in object detection and representation using visual data, primitives-based object representation remains a persistent challenge in computer vision. Consequently, high-quality spatial maps often rely on labor-intensive and manual processes. This paper introduces a novel deep learning methodology leveraging Graph Convolutional Networks (GCNs) to address these challenges in building footprint reconstruction. The proposed approach enhances performance by incorporating geometric regularity into building boundaries, integrating multi-scale and multi-resolution features, and embedding Attraction Field Maps into the network. These innovations provide a scalable and precise solution for automated building footprint extraction from a single satellite image, paving the way for impactful applications in urban planning, disaster management, and large-scale spatial analysis. Our model, Decoupled-PolyGCN, outperforms existing methods by 6% in AP and 10% in AR, demonstrating its ability to deliver accurate and regularized building footprints across diverse and challenging scenarios.

---

## Large Learning Rates Simultaneously Achieve Robustness to Spurious Correlations and Compressibility

**Link:** [https://arxiv.org/pdf/2507.17748](https://arxiv.org/pdf/2507.17748)

**Similarity:** 0.3322

### Abstract

Robustness and resource-efficiency are two highly desirable properties for modern machine learning models. However, achieving them jointly remains a challenge. In this paper, we position high learning rates as a facilitator for simultaneously achieving robustness to spurious correlations and network compressibility. We demonstrate that large learning rates also produce desirable representation properties such as invariant feature utilization, class separation, and activation sparsity. Importantly, our findings indicate that large learning rates compare favorably to other hyperparameters and regularization methods, in consistently satisfying these properties in tandem. In addition to demonstrating the positive effect of large learning rates across diverse spurious correlation datasets, models, and optimizers, we also present strong evidence that the previously documented success of large learning rates in standard classification tasks is likely due to its effect on addressing hidden/rare spurious correlations in the training dataset.

---

## Illicit object detection in X-ray imaging using deep learning techniques: A comparative evaluation

**Link:** [https://arxiv.org/pdf/2507.17508](https://arxiv.org/pdf/2507.17508)

**Similarity:** 0.3153

### Abstract

Automated X-ray inspection is crucial for efficient and unobtrusive security screening in various public settings. However, challenges such as object occlusion, variations in the physical properties of items, diversity in X-ray scanning devices, and limited training data hinder accurate and reliable detection of illicit items. Despite the large body of research in the field, reported experimental evaluations are often incomplete, with frequently conflicting outcomes. To shed light on the research landscape and facilitate further research, a systematic, detailed, and thorough comparative evaluation of recent Deep Learning (DL)-based methods for X-ray object detection is conducted. For this, a comprehensive evaluation framework is developed, composed of: a) Six recent, large-scale, and widely used public datasets for X-ray illicit item detection (OPIXray, CLCXray, SIXray, EDS, HiXray, and PIDray), b) Ten different state-of-the-art object detection schemes covering all main categories in the literature, including generic Convolutional Neural Network (CNN), custom CNN, generic transformer, and hybrid CNN-transformer architectures, and c) Various detection (mAP50 and mAP50:95) and time/computational-complexity (inference time (ms), parameter size (M), and computational load (GFLOPS)) metrics. A thorough analysis of the results leads to critical observations and insights, emphasizing key aspects such as: a) Overall behavior of the object detection schemes, b) Object-level detection performance, c) Dataset-specific observations, and d) Time efficiency and computational complexity analysis. To support reproducibility of the reported experimental results, the evaluation code and model weights are made publicly available at this https URL.

---

## Probing Vision-Language Understanding through the Visual Entailment Task: promises and pitfalls

**Link:** [https://arxiv.org/pdf/2507.17467](https://arxiv.org/pdf/2507.17467)

**Similarity:** 0.3143

### Abstract

This study investigates the extent to which the Visual Entailment (VE) task serves as a reliable probe of vision-language understanding in multimodal language models, using the LLaMA 3.2 11B Vision model as a test case. Beyond reporting performance metrics, we aim to interpret what these results reveal about the underlying possibilities and limitations of the VE task. We conduct a series of experiments across zero-shot, few-shot, and fine-tuning settings, exploring how factors such as prompt design, the number and order of in-context examples and access to visual information might affect VE performance. To further probe the reasoning processes of the model, we used explanation-based evaluations. Results indicate that three-shot inference outperforms the zero-shot baselines. However, additional examples introduce more noise than they provide benefits. Additionally, the order of the labels in the prompt is a critical factor that influences the predictions. In the absence of visual information, the model has a strong tendency to hallucinate and imagine content, raising questions about the model's over-reliance on linguistic priors. Fine-tuning yields strong results, achieving an accuracy of 83.3% on the e-SNLI-VE dataset and outperforming the state-of-the-art OFA-X model. Additionally, the explanation evaluation demonstrates that the fine-tuned model provides semantically meaningful explanations similar to those of humans, with a BERTScore F1-score of 89.2%. We do, however, find comparable BERTScore results in experiments with limited vision, questioning the visual grounding of this task. Overall, our results highlight both the utility and limitations of VE as a diagnostic task for vision-language understanding and point to directions for refining multimodal evaluation methods.

---

## Audio-Vision Contrastive Learning for Phonological Class Recognition

**Link:** [https://arxiv.org/pdf/2507.17682](https://arxiv.org/pdf/2507.17682)

**Similarity:** 0.3131

### Abstract

Accurate classification of articulatory-phonological features plays a vital role in understanding human speech production and developing robust speech technologies, particularly in clinical contexts where targeted phonemic analysis and therapy can improve disease diagnosis accuracy and personalized rehabilitation. In this work, we propose a multimodal deep learning framework that combines real-time magnetic resonance imaging (rtMRI) and speech signals to classify three key articulatory dimensions: manner of articulation, place of articulation, and voicing. We perform classification on 15 phonological classes derived from the aforementioned articulatory dimensions and evaluate the system with four audio/vision configurations: unimodal rtMRI, unimodal audio signals, multimodal middle fusion, and contrastive learning-based audio-vision fusion. Experimental results on the USC-TIMIT dataset show that our contrastive learning-based approach achieves state-of-the-art performance, with an average F1-score of 0.81, representing an absolute increase of 0.23 over the unimodal baseline. The results confirm the effectiveness of contrastive representation learning for multimodal articulatory analysis. Our code and processed dataset will be made publicly available at this https URL to support future research.

---

## Robust Five-Class and binary Diabetic Retinopathy Classification Using Transfer Learning and Data Augmentation

**Link:** [https://arxiv.org/pdf/2507.17121](https://arxiv.org/pdf/2507.17121)

**Similarity:** 0.3091

### Abstract

Diabetic retinopathy (DR) is a leading cause of vision loss worldwide, and early diagnosis through automated retinal image analysis can significantly reduce the risk of blindness. This paper presents a robust deep learning framework for both binary and five-class DR classification, leveraging transfer learning and extensive data augmentation to address the challenges of class imbalance and limited training data. We evaluate a range of pretrained convolutional neural network architectures, including variants of ResNet and EfficientNet, on the APTOS 2019 dataset.
For binary classification, our proposed model achieves a state-of-the-art accuracy of 98.9%, with a precision of 98.6%, recall of 99.3%, F1-score of 98.9%, and an AUC of 99.4%. In the more challenging five-class severity classification task, our model obtains a competitive accuracy of 84.6% and an AUC of 94.1%, outperforming several existing approaches. Our findings also demonstrate that EfficientNet-B0 and ResNet34 offer optimal trade-offs between accuracy and computational efficiency across both tasks.
These results underscore the effectiveness of combining class-balanced augmentation with transfer learning for high-performance DR diagnosis. The proposed framework provides a scalable and accurate solution for DR screening, with potential for deployment in real-world clinical environments.

---

## Multi-Scale PCB Defect Detection with YOLOv8 Network Improved via Pruning and Lightweight Network

**Link:** [https://arxiv.org/pdf/2507.17176](https://arxiv.org/pdf/2507.17176)

**Similarity:** 0.3034

### Abstract

With the high density of printed circuit board (PCB) design and the high speed of production, the traditional PCB defect detection model is difficult to take into account the accuracy and computational cost, and cannot meet the requirements of high accuracy and real-time detection of tiny defects. Therefore, in this paper, a multi-scale PCB defect detection method is improved with YOLOv8 using a comprehensive strategy of tiny target sensitivity strategy, network lightweighting and adaptive pruning, which is able to improve the detection speed and accuracy by optimizing the backbone network, the neck network and the detection head, the loss function and the adaptive pruning rate. Firstly, a Ghost-HGNetv2 structure with fewer parameters is used in the backbone network, and multilevel features are used to extract image semantic features to discover accurate defects. Secondly, we integrate C2f-Faster with small number of parameters in the neck section to enhance the ability of multi-level feature fusion. Next, in the Head part, we design a new GCDetect detection head, which allows the prediction of bounding boxes and categories to share the weights of GroupConv, and uses a small number of grouping convolutions to accomplish the regression and classification tasks, which significantly reduces the number of parameters while maintaining the accuracy of detection. We also design the Inner-MPDIoU boundary loss function to improve the detection and localization of tiny targets. Finally, the model was pruned by an optimized adaptive pruning rate to further reduce the complexity of the model. Experimental results show that the model exhibits advantages in terms of accuracy and speed. On the publicly available PCB defect dataset, mAP0.5 reaches 99.32% and mAP0.5:0.9 reaches 75.18%, which is 10.13% higher compared to YOLOv8n.

---

## CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation

**Link:** [https://arxiv.org/pdf/2507.17727](https://arxiv.org/pdf/2507.17727)

**Similarity:** 0.2993

### Abstract

State-of-the-art visual under-canopy navigation methods are designed with deep learning-based perception models to distinguish traversable space from crop rows. While these models have demonstrated successful performance, they require large amounts of training data to ensure reliability in real-world field deployment. However, data collection is costly, demanding significant human resources for in-field sampling and annotation. To address this challenge, various data augmentation techniques are commonly employed during model training, such as color jittering, Gaussian blur, and horizontal flip, to diversify training data and enhance model robustness. In this paper, we hypothesize that utilizing only these augmentation techniques may lead to suboptimal performance, particularly in complex under-canopy environments with frequent occlusions, debris, and non-uniform spacing of crops. Instead, we propose a novel augmentation method, so-called Crop-Aligned Cutout (CA-Cut) which masks random regions out in input images that are spatially distributed around crop rows on the sides to encourage trained models to capture high-level contextual features even when fine-grained information is obstructed. Our extensive experiments with a public cornfield dataset demonstrate that masking-based augmentations are effective for simulating occlusions and significantly improving robustness in semantic keypoint predictions for visual navigation. In particular, we show that biasing the mask distribution toward crop rows in CA-Cut is critical for enhancing both prediction accuracy and generalizability across diverse environments achieving up to a 36.9% reduction in prediction error. In addition, we conduct ablation studies to determine the number of masks, the size of each mask, and the spatial distribution of masks to maximize overall performance.

---

## Learning-based Stage Verification System in Manual Assembly Scenarios

**Link:** [https://arxiv.org/pdf/2507.17304](https://arxiv.org/pdf/2507.17304)

**Similarity:** 0.2829

### Abstract

In the context of Industry 4.0, effective monitoring of multiple targets and states during assembly processes is crucial, particularly when constrained to using only visual sensors. Traditional methods often rely on either multiple sensor types or complex hardware setups to achieve high accuracy in monitoring, which can be cost-prohibitive and difficult to implement in dynamic industrial environments. This study presents a novel approach that leverages multiple machine learning models to achieve precise monitoring under the limitation of using a minimal number of visual sensors. By integrating state information from identical timestamps, our method detects and confirms the current stage of the assembly process with an average accuracy exceeding 92%. Furthermore, our approach surpasses conventional methods by offering enhanced error detection and visuali-zation capabilities, providing real-time, actionable guidance to operators. This not only improves the accuracy and efficiency of assembly monitoring but also re-duces dependency on expensive hardware solutions, making it a more practical choice for modern industrial applications.

---

## Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts

**Link:** [https://arxiv.org/pdf/2507.16946](https://arxiv.org/pdf/2507.16946)

**Similarity:** 0.2826

### Abstract

Anomaly detection (AD) identifies the defect regions of a given image. Recent works have studied AD, focusing on learning AD without abnormal images, with long-tailed distributed training data, and using a unified model for all classes. In addition, online AD learning has also been explored. In this work, we expand in both directions to a realistic setting by considering the novel task of long-tailed online AD (LTOAD). We first identified that the offline state-of-the-art LTAD methods cannot be directly applied to the online setting. Specifically, LTAD is class-aware, requiring class labels that are not available in the online setting. To address this challenge, we propose a class-agnostic framework for LTAD and then adapt it to our online learning setting. Our method outperforms the SOTA baselines in most offline LTAD settings, including both the industrial manufacturing and the medical domain. In particular, we observe +4.63% image-AUROC on MVTec even compared to methods that have access to class labels and the number of classes. In the most challenging long-tailed online setting, we achieve +0.53% image-AUROC compared to baselines. Our LTOAD benchmark is released here: this https URL .

---

## DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD

**Link:** [https://arxiv.org/pdf/2507.17501](https://arxiv.org/pdf/2507.17501)

**Similarity:** 0.2821

### Abstract

Transformers have become the de facto backbone of modern deep learning, yet their training typically demands an advanced optimizer with adaptive learning rate like AdamW, rather than a momentum SGDW (mSGDW). Previous works show that it is mainly due to a heavy-tailed distribution of the gradients. In this paper, we introduce a Deeply Normalized Transformer (DNT), which is meticulously engineered to overcome this limitation enabling seamless training with vanilla mSGDW while yielding comparable performance to the Transformers trained via AdamW. To be specific, in DNT, we strategically integrate normalization techniques at proper positions in the Transformers to effectively modulate the Jacobian matrices of each layer, balance the influence of weights, activations, and their interactions, and thus enable the distributions of gradients concentrated. We provide both theoretical justifications of the normalization technique used in our DNT and extensive empirical evaluation on two popular Transformer architectures to validate that: a) DNT outperforms its counterparts (\ie, ViT and GPT), and b) DNT can be effectively trained with vanilla mSGDW.

---

## Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors

**Link:** [https://arxiv.org/pdf/2507.17577](https://arxiv.org/pdf/2507.17577)

**Similarity:** 0.2761

### Abstract

One of the most practical and challenging types of black-box adversarial attacks is the hard-label attack, where only the top-1 predicted label is available. One effective approach is to search for the optimal ray direction from the benign image that minimizes the $\ell_p$-norm distance to the adversarial region. The unique advantage of this approach is that it transforms the hard-label attack into a continuous optimization problem. The objective function value is the ray's radius, which can be obtained via binary search at a high query cost. Existing methods use a "sign trick" in gradient estimation to reduce the number of queries. In this paper, we theoretically analyze the quality of this gradient estimation and propose a novel prior-guided approach to improve ray search efficiency both theoretically and empirically. Specifically, we utilize the transfer-based priors from surrogate models, and our gradient estimators appropriately integrate them by approximating the projection of the true gradient onto the subspace spanned by these priors and random directions, in a query-efficient manner. We theoretically derive the expected cosine similarities between the obtained gradient estimators and the true gradient, and demonstrate the improvement achieved by incorporating priors. Extensive experiments on the ImageNet and CIFAR-10 datasets show that our approach significantly outperforms 11 state-of-the-art methods in terms of query efficiency.

---

## Exploring Active Learning for Semiconductor Defect Segmentation

**Link:** [https://arxiv.org/pdf/2507.17359](https://arxiv.org/pdf/2507.17359)

**Similarity:** 0.2749

### Abstract

The development of X-Ray microscopy (XRM) technology has enabled non-destructive inspection of semiconductor structures for defect identification. Deep learning is widely used as the state-of-the-art approach to perform visual analysis tasks. However, deep learning based models require large amount of annotated data to train. This can be time-consuming and expensive to obtain especially for dense prediction tasks like semantic segmentation. In this work, we explore active learning (AL) as a potential solution to alleviate the annotation burden. We identify two unique challenges when applying AL on semiconductor XRM scans: large domain shift and severe class-imbalance. To address these challenges, we propose to perform contrastive pretraining on the unlabelled data to obtain the initialization weights for each AL cycle, and a rareness-aware acquisition function that favors the selection of samples containing rare classes. We evaluate our method on a semiconductor dataset that is compiled from XRM scans of high bandwidth memory structures composed of logic and memory dies, and demonstrate that our method achieves state-of-the-art performance.

---

## Exploring Active Learning for Label-Efficient Training of Semantic Neural Radiance Field

**Link:** [https://arxiv.org/pdf/2507.17351](https://arxiv.org/pdf/2507.17351)

**Similarity:** 0.2694

### Abstract

Neural Radiance Field (NeRF) models are implicit neural scene representation methods that offer unprecedented capabilities in novel view synthesis. Semantically-aware NeRFs not only capture the shape and radiance of a scene, but also encode semantic information of the scene. The training of semantically-aware NeRFs typically requires pixel-level class labels, which can be prohibitively expensive to collect. In this work, we explore active learning as a potential solution to alleviate the annotation burden. We investigate various design choices for active learning of semantically-aware NeRF, including selection granularity and selection strategies. We further propose a novel active learning strategy that takes into account 3D geometric constraints in sample selection. Our experiments demonstrate that active learning can effectively reduce the annotation cost of training semantically-aware NeRF, achieving more than 2X reduction in annotation cost compared to random sampling.

---

## Swin-TUNA : A Novel PEFT Approach for Accurate Food Image Segmentation

**Link:** [https://arxiv.org/pdf/2507.17347](https://arxiv.org/pdf/2507.17347)

**Similarity:** 0.2671

### Abstract

In the field of food image processing, efficient semantic segmentation techniques are crucial for industrial applications. However, existing large-scale Transformer-based models (such as FoodSAM) face challenges in meeting practical deploymentrequirements due to their massive parameter counts and high computational resource demands. This paper introduces TUNable Adapter module (Swin-TUNA), a Parameter Efficient Fine-Tuning (PEFT) method that integrates multiscale trainable adapters into the Swin Transformer architecture, achieving high-performance food image segmentation by updating only 4% of the parameters. The core innovation of Swin-TUNA lies in its hierarchical feature adaptation mechanism: it designs separable convolutions in depth and dimensional mappings of varying scales to address the differences in features between shallow and deep networks, combined with a dynamic balancing strategy for tasks-agnostic and task-specific features. Experiments demonstrate that this method achieves mIoU of 50.56% and 74.94% on the FoodSeg103 and UECFoodPix Complete datasets, respectively, surpassing the fully parameterized FoodSAM model while reducing the parameter count by 98.7% (to only 8.13M). Furthermore, Swin-TUNA exhibits faster convergence and stronger generalization capabilities in low-data scenarios, providing an efficient solution for assembling lightweight food image.

---

## RemixFusion: Residual-based Mixed Representation for Large-scale Online RGB-D Reconstruction

**Link:** [https://arxiv.org/pdf/2507.17594](https://arxiv.org/pdf/2507.17594)

**Similarity:** 0.2654

### Abstract

The introduction of the neural implicit representation has notably propelled the advancement of online dense reconstruction techniques. Compared to traditional explicit representations, such as TSDF, it improves the mapping completeness and memory efficiency. However, the lack of reconstruction details and the time-consuming learning of neural representations hinder the widespread application of neural-based methods to large-scale online reconstruction. We introduce RemixFusion, a novel residual-based mixed representation for scene reconstruction and camera pose estimation dedicated to high-quality and large-scale online RGB-D reconstruction. In particular, we propose a residual-based map representation comprised of an explicit coarse TSDF grid and an implicit neural module that produces residuals representing fine-grained details to be added to the coarse grid. Such mixed representation allows for detail-rich reconstruction with bounded time and memory budget, contrasting with the overly-smoothed results by the purely implicit representations, thus paving the way for high-quality camera tracking. Furthermore, we extend the residual-based representation to handle multi-frame joint pose optimization via bundle adjustment (BA). In contrast to the existing methods, which optimize poses directly, we opt to optimize pose changes. Combined with a novel technique for adaptive gradient amplification, our method attains better optimization convergence and global optimality. Furthermore, we adopt a local moving volume to factorize the mixed scene representation with a divide-and-conquer design to facilitate efficient online learning in our residual-based framework. Extensive experiments demonstrate that our method surpasses all state-of-the-art ones, including those based either on explicit or implicit representations, in terms of the accuracy of both mapping and tracking on large-scale scenes.

---

## SRMambaV2: Biomimetic Attention for Sparse Point Cloud Upsampling in Autonomous Driving

**Link:** [https://arxiv.org/pdf/2507.17479](https://arxiv.org/pdf/2507.17479)

**Similarity:** 0.2621

### Abstract

Upsampling LiDAR point clouds in autonomous driving scenarios remains a significant challenge due to the inherent sparsity and complex 3D structures of the data. Recent studies have attempted to address this problem by converting the complex 3D spatial scenes into 2D image super-resolution tasks. However, due to the sparse and blurry feature representation of range images, accurately reconstructing detailed and complex spatial topologies remains a major difficulty. To tackle this, we propose a novel sparse point cloud upsampling method named SRMambaV2, which enhances the upsampling accuracy in long-range sparse regions while preserving the overall geometric reconstruction quality. Specifically, inspired by human driver visual perception, we design a biomimetic 2D selective scanning self-attention (2DSSA) mechanism to model the feature distribution in distant sparse areas. Meanwhile, we introduce a dual-branch network architecture to enhance the representation of sparse features. In addition, we introduce a progressive adaptive loss (PAL) function to further refine the reconstruction of fine-grained details during the upsampling process. Experimental results demonstrate that SRMambaV2 achieves superior performance in both qualitative and quantitative evaluations, highlighting its effectiveness and practical value in automotive sparse point cloud upsampling tasks.

---

## On the Interaction of Compressibility and Adversarial Robustness

**Link:** [https://arxiv.org/pdf/2507.17725](https://arxiv.org/pdf/2507.17725)

**Similarity:** 0.2601

### Abstract

Modern neural networks are expected to simultaneously satisfy a host of desirable properties: accurate fitting to training data, generalization to unseen inputs, parameter and computational efficiency, and robustness to adversarial perturbations. While compressibility and robustness have each been studied extensively, a unified understanding of their interaction still remains elusive. In this work, we develop a principled framework to analyze how different forms of compressibility - such as neuron-level sparsity and spectral compressibility - affect adversarial robustness. We show that these forms of compression can induce a small number of highly sensitive directions in the representation space, which adversaries can exploit to construct effective perturbations. Our analysis yields a simple yet instructive robustness bound, revealing how neuron and spectral compressibility impact $L_\infty$ and $L_2$ robustness via their effects on the learned representations. Crucially, the vulnerabilities we identify arise irrespective of how compression is achieved - whether via regularization, architectural bias, or implicit learning dynamics. Through empirical evaluations across synthetic and realistic tasks, we confirm our theoretical predictions, and further demonstrate that these vulnerabilities persist under adversarial training and transfer learning, and contribute to the emergence of universal adversarial perturbations. Our findings show a fundamental tension between structured compressibility and robustness, and suggest new pathways for designing models that are both efficient and secure.

---

## A Low-Cost Machine Learning Approach for Timber Diameter Estimation

**Link:** [https://arxiv.org/pdf/2507.17219](https://arxiv.org/pdf/2507.17219)

**Similarity:** 0.2584

### Abstract

The wood processing industry, particularly in facilities such as sawmills and MDF production lines, requires accurate and efficient identification of species and thickness of the wood. Although traditional methods rely heavily on expert human labor, they are slow, inconsistent, and prone to error, especially when processing large volumes. This study focuses on practical and cost-effective machine learning frameworks that automate the estimation of timber log diameter using standard RGB images captured under real-world working conditions. We employ the YOLOv5 object detection algorithm, fine-tuned on a public dataset (TimberSeg 1.0), to detect individual timber logs and estimate thickness through bounding-box dimensions. Unlike previous methods that require expensive sensors or controlled environments, this model is trained on images taken in typical industrial sheds during timber delivery. Experimental results show that the model achieves a mean Average Precision (mAP@0.5) of 0.64, demonstrating reliable log detection even with modest computing resources. This lightweight, scalable solution holds promise for practical integration into existing workflows, including on-site inventory management and preliminary sorting, particularly in small and medium-sized operations.

---

## DFDNet: Dynamic Frequency-Guided De-Flare Network

**Link:** [https://arxiv.org/pdf/2507.17489](https://arxiv.org/pdf/2507.17489)

**Similarity:** 0.2567

### Abstract

Strong light sources in nighttime photography frequently produce flares in images, significantly degrading visual quality and impacting the performance of downstream tasks. While some progress has been made, existing methods continue to struggle with removing large-scale flare artifacts and repairing structural damage in regions near the light source. We observe that these challenging flare artifacts exhibit more significant discrepancies from the reference images in the frequency domain compared to the spatial domain. Therefore, this paper presents a novel dynamic frequency-guided deflare network (DFDNet) that decouples content information from flare artifacts in the frequency domain, effectively removing large-scale flare artifacts. Specifically, DFDNet consists mainly of a global dynamic frequency-domain guidance (GDFG) module and a local detail guidance module (LDGM). The GDFG module guides the network to perceive the frequency characteristics of flare artifacts by dynamically optimizing global frequency domain features, effectively separating flare information from content information. Additionally, we design an LDGM via a contrastive learning strategy that aligns the local features of the light source with the reference image, reduces local detail damage from flare removal, and improves fine-grained image restoration. The experimental results demonstrate that the proposed method outperforms existing state-of-the-art methods in terms of performance. The code is available at \href{this https URL}{this https URL}.

---

## MyGO: Make your Goals Obvious, Avoiding Semantic Confusion in Prostate Cancer Lesion Region Segmentation

**Link:** [https://arxiv.org/pdf/2507.17269](https://arxiv.org/pdf/2507.17269)

**Similarity:** 0.2562

### Abstract

Early diagnosis and accurate identification of lesion location and progression in prostate cancer (PCa) are critical for assisting clinicians in formulating effective treatment strategies. However, due to the high semantic homogeneity between lesion and non-lesion areas, existing medical image segmentation methods often struggle to accurately comprehend lesion semantics, resulting in the problem of semantic confusion. To address this challenge, we propose a novel Pixel Anchor Module, which guides the model to discover a sparse set of feature anchors that serve to capture and interpret global contextual information. This mechanism enhances the model's nonlinear representation capacity and improves segmentation accuracy within lesion regions. Moreover, we design a self-attention-based Top_k selection strategy to further refine the identification of these feature anchors, and incorporate a focal loss function to mitigate class imbalance, thereby facilitating more precise semantic interpretation across diverse regions. Our method achieves state-of-the-art performance on the PI-CAI dataset, demonstrating 69.73% IoU and 74.32% Dice scores, and significantly improving prostate cancer lesion detection.

---

## Pixels, Patterns, but No Poetry: To See The World like Humans

**Link:** [https://arxiv.org/pdf/2507.16863](https://arxiv.org/pdf/2507.16863)

**Similarity:** 0.2557

### Abstract

Achieving human-like perception and reasoning in Multimodal Large Language Models (MLLMs) remains a central challenge in artificial intelligence. While recent research has primarily focused on enhancing reasoning capabilities in MLLMs, a fundamental question persists: Can Multimodal Large Language Models truly perceive the world as humans do? This paper shifts focus from reasoning to perception. Rather than constructing benchmarks specifically for reasoning, we introduce the Turing Eye Test (TET), a challenging perception-oriented benchmark comprising four diagnostic tasks that evaluate MLLMs' performance on synthetic images that humans process intuitively. Our findings reveal that state-of-the-art MLLMs exhibit catastrophic failures on our perceptual tasks trivial for humans. Both in-context learning and training on language backbone-effective for previous benchmarks-fail to improve performance on our tasks, while fine-tuning the vision tower enables rapid adaptation, suggesting that our benchmark poses challenges for vision tower generalization rather than for the knowledge and reasoning capabilities of the language backbone-a key gap between current MLLMs and human perception. We release a representative subset of TET tasks in this version, and will introduce more diverse tasks and methods to enhance visual generalization in future work.

---

## Controllable Hybrid Captioner for Improved Long-form Video Understanding

**Link:** [https://arxiv.org/pdf/2507.17047](https://arxiv.org/pdf/2507.17047)

**Similarity:** 0.2555

### Abstract

Video data, especially long-form video, is extremely dense and high-dimensional. Text-based summaries of video content offer a way to represent query-relevant content in a much more compact manner than raw video. In addition, textual representations are easily ingested by state-of-the-art large language models (LLMs), which enable reasoning over video content to answer complex natural language queries. To solve this issue, we rely on the progressive construction of a text-based memory by a video captioner operating on shorter chunks of the video, where spatio-temporal modeling is computationally feasible. We explore ways to improve the quality of the activity log comprised solely of short video captions. Because the video captions tend to be focused on human actions, and questions may pertain to other information in the scene, we seek to enrich the memory with static scene descriptions using Vision Language Models (VLMs). Our video understanding system relies on the LaViLa video captioner in combination with a LLM to answer questions about videos. We first explored different ways of partitioning the video into meaningful segments such that the textual descriptions more accurately reflect the structure of the video content. Furthermore, we incorporated static scene descriptions into the captioning pipeline using LLaVA VLM, resulting in a more detailed and complete caption log and expanding the space of questions that are answerable from the textual memory. Finally, we have successfully fine-tuned the LaViLa video captioner to produce both action and scene captions, significantly improving the efficiency of the captioning pipeline compared to using separate captioning models for the two tasks. Our model, controllable hybrid captioner, can alternate between different types of captions according to special input tokens that signals scene changes detected in the video.

---

## A Hybrid CNN-VSSM model for Multi-View, Multi-Task Mammography Analysis: Robust Diagnosis with Attention-Based Fusion

**Link:** [https://arxiv.org/pdf/2507.16955](https://arxiv.org/pdf/2507.16955)

**Similarity:** 0.2536

### Abstract

Early and accurate interpretation of screening mammograms is essential for effective breast cancer detection, yet it remains a complex challenge due to subtle imaging findings and diagnostic ambiguity. Many existing AI approaches fall short by focusing on single view inputs or single-task outputs, limiting their clinical utility. To address these limitations, we propose a novel multi-view, multitask hybrid deep learning framework that processes all four standard mammography views and jointly predicts diagnostic labels and BI-RADS scores for each breast. Our architecture integrates a hybrid CNN VSSM backbone, combining convolutional encoders for rich local feature extraction with Visual State Space Models (VSSMs) to capture global contextual dependencies. To improve robustness and interpretability, we incorporate a gated attention-based fusion module that dynamically weights information across views, effectively handling cases with missing data. We conduct extensive experiments across diagnostic tasks of varying complexity, benchmarking our proposed hybrid models against baseline CNN architectures and VSSM models in both single task and multi task learning settings. Across all tasks, the hybrid models consistently outperform the baselines. In the binary BI-RADS 1 vs. 5 classification task, the shared hybrid model achieves an AUC of 0.9967 and an F1 score of 0.9830. For the more challenging ternary classification, it attains an F1 score of 0.7790, while in the five-class BI-RADS task, the best F1 score reaches 0.4904. These results highlight the effectiveness of the proposed hybrid framework and underscore both the potential and limitations of multitask learning for improving diagnostic performance and enabling clinically meaningful mammography analysis.

---

## Asymmetric Lesion Detection with Geometric Patterns and CNN-SVM Classification

**Link:** [https://arxiv.org/pdf/2507.17185](https://arxiv.org/pdf/2507.17185)

**Similarity:** 0.2527

### Abstract

In dermoscopic images, which allow visualization of surface skin structures not visible to the naked eye, lesion shape offers vital insights into skin diseases. In clinically practiced methods, asymmetric lesion shape is one of the criteria for diagnosing melanoma. Initially, we labeled data for a non-annotated dataset with symmetrical information based on clinical assessments. Subsequently, we propose a supporting technique, a supervised learning image processing algorithm, to analyze the geometrical pattern of lesion shape, aiding non-experts in understanding the criteria of an asymmetric lesion. We then utilize a pre-trained convolutional neural network (CNN) to extract shape, color, and texture features from dermoscopic images for training a multiclass support vector machine (SVM) classifier, outperforming state-of-the-art methods from the literature. In the geometry-based experiment, we achieved a 99.00% detection rate for dermatological asymmetric lesions. In the CNN-based experiment, the best performance is found with 94% Kappa Score, 95% Macro F1-score, and 97% Weighted F1-score for classifying lesion shapes (Asymmetric, Half-Symmetric, and Symmetric).

---

## DeMo++: Motion Decoupling for Autonomous Driving

**Link:** [https://arxiv.org/pdf/2507.17342](https://arxiv.org/pdf/2507.17342)

**Similarity:** 0.2502

### Abstract

Motion forecasting and planning are tasked with estimating the trajectories of traffic agents and the ego vehicle, respectively, to ensure the safety and efficiency of autonomous driving systems in dynamically changing environments. State-of-the-art methods typically adopt a one-query-one-trajectory paradigm, where each query corresponds to a unique trajectory for predicting multi-mode trajectories. While this paradigm can produce diverse motion intentions, it often falls short in modeling the intricate spatiotemporal evolution of trajectories, which can lead to collisions or suboptimal outcomes. To overcome this limitation, we propose DeMo++, a framework that decouples motion estimation into two distinct components: holistic motion intentions to capture the diverse potential directions of movement, and fine spatiotemporal states to track the agent's dynamic progress within the scene and enable a self-refinement capability. Further, we introduce a cross-scene trajectory interaction mechanism to explore the relationships between motions in adjacent scenes. This allows DeMo++ to comprehensively model both the diversity of motion intentions and the spatiotemporal evolution of each trajectory. To effectively implement this framework, we developed a hybrid model combining Attention and Mamba. This architecture leverages the strengths of both mechanisms for efficient scene information aggregation and precise trajectory state sequence modeling. Extensive experiments demonstrate that DeMo++ achieves state-of-the-art performance across various benchmarks, including motion forecasting (Argoverse 2 and nuScenes), motion planning (nuPlan), and end-to-end planning (NAVSIM).

---

## Multi-modal Multi-task Pre-training for Improved Point Cloud Understanding

**Link:** [https://arxiv.org/pdf/2507.17533](https://arxiv.org/pdf/2507.17533)

**Similarity:** 0.2492

### Abstract

Recent advances in multi-modal pre-training methods have shown promising effectiveness in learning 3D representations by aligning multi-modal features between 3D shapes and their corresponding 2D counterparts. However, existing multi-modal pre-training frameworks primarily rely on a single pre-training task to gather multi-modal data in 3D applications. This limitation prevents the models from obtaining the abundant information provided by other relevant tasks, which can hinder their performance in downstream tasks, particularly in complex and diverse domains. In order to tackle this issue, we propose MMPT, a Multi-modal Multi-task Pre-training framework designed to enhance point cloud understanding. Specifically, three pre-training tasks are devised: (i) Token-level reconstruction (TLR) aims to recover masked point tokens, endowing the model with representative learning abilities. (ii) Point-level reconstruction (PLR) is integrated to predict the masked point positions directly, and the reconstructed point cloud can be considered as a transformed point cloud used in the subsequent task. (iii) Multi-modal contrastive learning (MCL) combines feature correspondences within and across modalities, thus assembling a rich learning signal from both 3D point cloud and 2D image modalities in a self-supervised manner. Moreover, this framework operates without requiring any 3D annotations, making it scalable for use with large datasets. The trained encoder can be effectively transferred to various downstream tasks. To demonstrate its effectiveness, we evaluated its performance compared to state-of-the-art methods in various discriminant and generative applications under widely-used benchmarks.

---

## A Conditional Probability Framework for Compositional Zero-shot Learning

**Link:** [https://arxiv.org/pdf/2507.17377](https://arxiv.org/pdf/2507.17377)

**Similarity:** 0.2487

### Abstract

Compositional Zero-Shot Learning (CZSL) aims to recognize unseen combinations of known objects and attributes by leveraging knowledge from previously seen compositions. Traditional approaches primarily focus on disentangling attributes and objects, treating them as independent entities during learning. However, this assumption overlooks the semantic constraints and contextual dependencies inside a composition. For example, certain attributes naturally pair with specific objects (e.g., "striped" applies to "zebra" or "shirts" but not "sky" or "water"), while the same attribute can manifest differently depending on context (e.g., "young" in "young tree" vs. "young dog"). Thus, capturing attribute-object interdependence remains a fundamental yet long-ignored challenge in CZSL. In this paper, we adopt a Conditional Probability Framework (CPF) to explicitly model attribute-object dependencies. We decompose the probability of a composition into two components: the likelihood of an object and the conditional likelihood of its attribute. To enhance object feature learning, we incorporate textual descriptors to highlight semantically relevant image regions. These enhanced object features then guide attribute learning through a cross-attention mechanism, ensuring better contextual alignment. By jointly optimizing object likelihood and conditional attribute likelihood, our method effectively captures compositional dependencies and generalizes well to unseen compositions. Extensive experiments on multiple CZSL benchmarks demonstrate the superiority of our approach. Code is available at here.

---

## Finding Dori: Memorization in Text-to-Image Diffusion Models Is Less Local Than Assumed

**Link:** [https://arxiv.org/pdf/2507.16880](https://arxiv.org/pdf/2507.16880)

**Similarity:** 0.2472

### Abstract

Text-to-image diffusion models (DMs) have achieved remarkable success in image generation. However, concerns about data privacy and intellectual property remain due to their potential to inadvertently memorize and replicate training data. Recent mitigation efforts have focused on identifying and pruning weights responsible for triggering replication, based on the assumption that memorization can be localized. Our research assesses the robustness of these pruning-based approaches. We demonstrate that even after pruning, minor adjustments to text embeddings of input prompts are sufficient to re-trigger data replication, highlighting the fragility of these defenses. Furthermore, we challenge the fundamental assumption of memorization locality, by showing that replication can be triggered from diverse locations within the text embedding space, and follows different paths in the model. Our findings indicate that existing mitigation strategies are insufficient and underscore the need for methods that truly remove memorized content, rather than attempting to suppress its retrieval. As a first step in this direction, we introduce a novel adversarial fine-tuning method that iteratively searches for replication triggers and updates the model to increase robustness. Through our research, we provide fresh insights into the nature of memorization in text-to-image DMs and a foundation for building more trustworthy and compliant generative AI.

---

## Exploring Spatial Diversity for Region-based Active Learning

**Link:** [https://arxiv.org/pdf/2507.17367](https://arxiv.org/pdf/2507.17367)

**Similarity:** 0.2469

### Abstract

State-of-the-art methods for semantic segmentation are based on deep neural networks trained on large-scale labeled datasets. Acquiring such datasets would incur large annotation costs, especially for dense pixel-level prediction tasks like semantic segmentation. We consider region-based active learning as a strategy to reduce annotation costs while maintaining high performance. In this setting, batches of informative image regions instead of entire images are selected for labeling. Importantly, we propose that enforcing local spatial diversity is beneficial for active learning in this case, and to incorporate spatial diversity along with the traditional active selection criterion, e.g., data sample uncertainty, in a unified optimization framework for region-based active learning. We apply this framework to the Cityscapes and PASCAL VOC datasets and demonstrate that the inclusion of spatial diversity effectively improves the performance of uncertainty-based and feature diversity-based active learning methods. Our framework achieves $95\%$ performance of fully supervised methods with only $5-9\%$ of the labeled pixels, outperforming all state-of-the-art region-based active learning methods for semantic segmentation.

---

## Unsupervised Exposure Correction

**Link:** [https://arxiv.org/pdf/2507.17252](https://arxiv.org/pdf/2507.17252)

**Similarity:** 0.2462

### Abstract

Current exposure correction methods have three challenges, labor-intensive paired data annotation, limited generalizability, and performance degradation in low-level computer vision tasks. In this work, we introduce an innovative Unsupervised Exposure Correction (UEC) method that eliminates the need for manual annotations, offers improved generalizability, and enhances performance in low-level downstream tasks. Our model is trained using freely available paired data from an emulated Image Signal Processing (ISP) pipeline. This approach does not need expensive manual annotations, thereby minimizing individual style biases from the annotation and consequently improving its generalizability. Furthermore, we present a large-scale Radiometry Correction Dataset, specifically designed to emphasize exposure variations, to facilitate unsupervised learning. In addition, we develop a transformation function that preserves image details and outperforms state-of-the-art supervised methods [12], while utilizing only 0.01% of their parameters. Our work further investigates the broader impact of exposure correction on downstream tasks, including edge detection, demonstrating its effectiveness in mitigating the adverse effects of poor exposure on low-level features. The source code and dataset are publicly available at this https URL.

---

## Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning

**Link:** [https://arxiv.org/pdf/2507.17539](https://arxiv.org/pdf/2507.17539)

**Similarity:** 0.2454

### Abstract

Multimodal large language models (MLLMs) demonstrate significant potential in the field of medical diagnosis. However, they face critical challenges in specialized domains such as ophthalmology, particularly the fragmentation of annotation granularity and inconsistencies in clinical reasoning logic, which hinder precise cross-modal understanding. This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system. Fundus-Engine automates localization and leverages MLLM-based semantic expansion to integrate global disease classification, local object detection, and fine-grained feature analysis within a single fundus image. Additionally, by constructing a clinically aligned cognitive chain, it guides the model to generate interpretable reasoning paths. FundusExpert, fine-tuned with instruction data from FundusGen, achieves the best performance in ophthalmic question-answering tasks, surpassing the average accuracy of the 40B MedRegA by 26.6%. It also excels in zero-shot report generation tasks, achieving a clinical consistency of 77.0%, significantly outperforming GPT-4o's 47.6%. Furthermore, we reveal a scaling law between data quality and model capability ($L \propto N^{0.068}$), demonstrating that the cognitive alignment annotations in FundusGen enhance data utilization efficiency. By integrating region-level localization with diagnostic reasoning chains, our work develops a scalable, clinically-aligned MLLM and explores a pathway toward bridging the visual-language gap in specific MLLMs. Our project can be found at this https URL.

---

## CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance

**Link:** [https://arxiv.org/pdf/2507.17312](https://arxiv.org/pdf/2507.17312)

**Similarity:** 0.2447

### Abstract

Semi-dense feature matching methods have shown strong performance in challenging scenarios. However, the existing pipeline relies on a global search across the entire feature map to establish coarse matches, limiting further improvements in accuracy and efficiency. Motivated by this limitation, we propose a novel pipeline, CasP, which leverages cascaded correspondence priors for guidance. Specifically, the matching stage is decomposed into two progressive phases, bridged by a region-based selective cross-attention mechanism designed to enhance feature discriminability. In the second phase, one-to-one matches are determined by restricting the search range to the one-to-many prior areas identified in the first phase. Additionally, this pipeline benefits from incorporating high-level features, which helps reduce the computational costs of low-level feature extraction. The acceleration gains of CasP increase with higher resolution, and our lite model achieves a speedup of $\sim2.2\times$ at a resolution of 1152 compared to the most efficient method, ELoFTR. Furthermore, extensive experiments demonstrate its superiority in geometric estimation, particularly with impressive cross-domain generalization. These advantages highlight its potential for latency-sensitive and high-robustness applications, such as SLAM and UAV systems. Code is available at this https URL.

---

## ScSAM: Debiasing Morphology and Distributional Variability in Subcellular Semantic Segmentation

**Link:** [https://arxiv.org/pdf/2507.17149](https://arxiv.org/pdf/2507.17149)

**Similarity:** 0.2442

### Abstract

The significant morphological and distributional variability among subcellular components poses a long-standing challenge for learning-based organelle segmentation models, significantly increasing the risk of biased feature learning. Existing methods often rely on single mapping relationships, overlooking feature diversity and thereby inducing biased training. Although the Segment Anything Model (SAM) provides rich feature representations, its application to subcellular scenarios is hindered by two key challenges: (1) The variability in subcellular morphology and distribution creates gaps in the label space, leading the model to learn spurious or biased features. (2) SAM focuses on global contextual understanding and often ignores fine-grained spatial details, making it challenging to capture subtle structural alterations and cope with skewed data distributions. To address these challenges, we introduce ScSAM, a method that enhances feature robustness by fusing pre-trained SAM with Masked Autoencoder (MAE)-guided cellular prior knowledge to alleviate training bias from data imbalance. Specifically, we design a feature alignment and fusion module to align pre-trained embeddings to the same feature space and efficiently combine different representations. Moreover, we present a cosine similarity matrix-based class prompt encoder to activate class-specific features to recognize subcellular categories. Extensive experiments on diverse subcellular image datasets demonstrate that ScSAM outperforms state-of-the-art methods.

---

## A tissue and cell-level annotated H&E and PD-L1 histopathology image dataset in non-small cell lung cancer

**Link:** [https://arxiv.org/pdf/2507.16855](https://arxiv.org/pdf/2507.16855)

**Similarity:** 0.2432

### Abstract

The tumor immune microenvironment (TIME) in non-small cell lung cancer (NSCLC) histopathology contains morphological and molecular characteristics predictive of immunotherapy response. Computational quantification of TIME characteristics, such as cell detection and tissue segmentation, can support biomarker development. However, currently available digital pathology datasets of NSCLC for the development of cell detection or tissue segmentation algorithms are limited in scope, lack annotations of clinically prevalent metastatic sites, and forgo molecular information such as PD-L1 immunohistochemistry (IHC). To fill this gap, we introduce the IGNITE data toolkit, a multi-stain, multi-centric, and multi-scanner dataset of annotated NSCLC whole-slide images. We publicly release 887 fully annotated regions of interest from 155 unique patients across three complementary tasks: (i) multi-class semantic segmentation of tissue compartments in H&E-stained slides, with 16 classes spanning primary and metastatic NSCLC, (ii) nuclei detection, and (iii) PD-L1 positive tumor cell detection in PD-L1 IHC slides. To the best of our knowledge, this is the first public NSCLC dataset with manual annotations of H&E in metastatic sites and PD-L1 IHC.

---

## Unsupervised anomaly detection using Bayesian flow networks: application to brain FDG PET in the context of Alzheimer's disease

**Link:** [https://arxiv.org/pdf/2507.17486](https://arxiv.org/pdf/2507.17486)

**Similarity:** 0.2427

### Abstract

Unsupervised anomaly detection (UAD) plays a crucial role in neuroimaging for identifying deviations from healthy subject data and thus facilitating the diagnosis of neurological disorders. In this work, we focus on Bayesian flow networks (BFNs), a novel class of generative models, which have not yet been applied to medical imaging or anomaly detection. BFNs combine the strength of diffusion frameworks and Bayesian inference. We introduce AnoBFN, an extension of BFNs for UAD, designed to: i) perform conditional image generation under high levels of spatially correlated noise, and ii) preserve subject specificity by incorporating a recursive feedback from the input image throughout the generative process. We evaluate AnoBFN on the challenging task of Alzheimer's disease-related anomaly detection in FDG PET images. Our approach outperforms other state-of-the-art methods based on VAEs (beta-VAE), GANs (f-AnoGAN), and diffusion models (AnoDDPM), demonstrating its effectiveness at detecting anomalies while reducing false positive rates.

---

## Mammo-Mamba: A Hybrid State-Space and Transformer Architecture with Sequential Mixture of Experts for Multi-View Mammography

**Link:** [https://arxiv.org/pdf/2507.17662](https://arxiv.org/pdf/2507.17662)

**Similarity:** 0.2421

### Abstract

Breast cancer (BC) remains one of the leading causes of cancer-related mortality among women, despite recent advances in Computer-Aided Diagnosis (CAD) systems. Accurate and efficient interpretation of multi-view mammograms is essential for early detection, driving a surge of interest in Artificial Intelligence (AI)-powered CAD models. While state-of-the-art multi-view mammogram classification models are largely based on Transformer architectures, their computational complexity scales quadratically with the number of image patches, highlighting the need for more efficient alternatives. To address this challenge, we propose Mammo-Mamba, a novel framework that integrates Selective State-Space Models (SSMs), transformer-based attention, and expert-driven feature refinement into a unified architecture. Mammo-Mamba extends the MambaVision backbone by introducing the Sequential Mixture of Experts (SeqMoE) mechanism through its customized SecMamba block. The SecMamba is a modified MambaVision block that enhances representation learning in high-resolution mammographic images by enabling content-adaptive feature refinement. These blocks are integrated into the deeper stages of MambaVision, allowing the model to progressively adjust feature emphasis through dynamic expert gating, effectively mitigating the limitations of traditional Transformer models. Evaluated on the CBIS-DDSM benchmark dataset, Mammo-Mamba achieves superior classification performance across all key metrics while maintaining computational efficiency.

---

## Perceptual Classifiers: Detecting Generative Images using Perceptual Features

**Link:** [https://arxiv.org/pdf/2507.17240](https://arxiv.org/pdf/2507.17240)

**Similarity:** 0.2412

### Abstract

Image Quality Assessment (IQA) models are employed in many practical image and video processing pipelines to reduce storage, minimize transmission costs, and improve the Quality of Experience (QoE) of millions of viewers. These models are sensitive to a diverse range of image distortions and can accurately predict image quality as judged by human viewers. Recent advancements in generative models have resulted in a significant influx of "GenAI" content on the internet. Existing methods for detecting GenAI content have progressed significantly with improved generalization performance on images from unseen generative models. Here, we leverage the capabilities of existing IQA models, which effectively capture the manifold of real images within a bandpass statistical space, to distinguish between real and AI-generated images. We investigate the generalization ability of these perceptual classifiers to the task of GenAI image detection and evaluate their robustness against various image degradations. Our results show that a two-layer network trained on the feature space of IQA models demonstrates state-of-the-art performance in detecting fake images across generative models, while maintaining significant robustness against image degradations.

---

## BetterCheck: Towards Safeguarding VLMs for Automotive Perception Systems

**Link:** [https://arxiv.org/pdf/2507.17722](https://arxiv.org/pdf/2507.17722)

**Similarity:** 0.2395

### Abstract

Large language models (LLMs) are growingly extended to process multimodal data such as text and video simultaneously. Their remarkable performance in understanding what is shown in images is surpassing specialized neural networks (NNs) such as Yolo that is supporting only a well-formed but very limited vocabulary, ie., objects that they are able to detect. When being non-restricted, LLMs and in particular state-of-the-art vision language models (VLMs) show impressive performance to describe even complex traffic situations. This is making them potentially suitable components for automotive perception systems to support the understanding of complex traffic situations or edge case situation. However, LLMs and VLMs are prone to hallucination, which mean to either potentially not seeing traffic agents such as vulnerable road users who are present in a situation, or to seeing traffic agents who are not there in reality. While the latter is unwanted making an ADAS or autonomous driving systems (ADS) to unnecessarily slow down, the former could lead to disastrous decisions from an ADS. In our work, we are systematically assessing the performance of 3 state-of-the-art VLMs on a diverse subset of traffic situations sampled from the Waymo Open Dataset to support safety guardrails for capturing such hallucinations in VLM-supported perception systems. We observe that both, proprietary and open VLMs exhibit remarkable image understanding capabilities even paying thorough attention to fine details sometimes difficult to spot for us humans. However, they are also still prone to making up elements in their descriptions to date requiring hallucination detection strategies such as BetterCheck that we propose in our work.

---

## ERMV: Editing 4D Robotic Multi-view images to enhance embodied agents

**Link:** [https://arxiv.org/pdf/2507.17462](https://arxiv.org/pdf/2507.17462)

**Similarity:** 0.2384

### Abstract

Robot imitation learning relies on 4D multi-view sequential images. However, the high cost of data collection and the scarcity of high-quality data severely constrain the generalization and application of embodied intelligence policies like Vision-Language-Action (VLA) models. Data augmentation is a powerful strategy to overcome data scarcity, but methods for editing 4D multi-view sequential images for manipulation tasks are currently lacking. Thus, we propose ERMV (Editing Robotic Multi-View 4D data), a novel data augmentation framework that efficiently edits an entire multi-view sequence based on single-frame editing and robot state conditions. This task presents three core challenges: (1) maintaining geometric and appearance consistency across dynamic views and long time horizons; (2) expanding the working window with low computational costs; and (3) ensuring the semantic integrity of critical objects like the robot arm. ERMV addresses these challenges through a series of innovations. First, to ensure spatio-temporal consistency in motion blur, we introduce a novel Epipolar Motion-Aware Attention (EMA-Attn) mechanism that learns pixel shift caused by movement before applying geometric constraints. Second, to maximize the editing working window, ERMV pioneers a Sparse Spatio-Temporal (STT) module, which decouples the temporal and spatial views and remodels a single-frame multi-view problem through sparse sampling of the views to reduce computational demands. Third, to alleviate error accumulation, we incorporate a feedback intervention Mechanism, which uses a Multimodal Large Language Model (MLLM) to check editing inconsistencies and request targeted expert guidance only when necessary. Extensive experiments demonstrate that ERMV-augmented data significantly boosts the robustness and generalization of VLA models in both simulated and real-world environments.

---

## Principled Multimodal Representation Learning

**Link:** [https://arxiv.org/pdf/2507.17343](https://arxiv.org/pdf/2507.17343)

**Similarity:** 0.2373

### Abstract

Multimodal representation learning seeks to create a unified representation space by integrating diverse data modalities to improve multimodal understanding. Traditional methods often depend on pairwise contrastive learning, which relies on a predefined anchor modality, restricting alignment across all modalities. Recent advances have investigated the simultaneous alignment of multiple modalities, yet several challenges remain, such as limitations imposed by fixed anchor points and instability arising from optimizing the product of singular values. To address the challenges, in this paper, we propose Principled Multimodal Representation Learning (PMRL), a novel framework that achieves simultaneous alignment of multiple modalities without anchor dependency in a more stable manner. Specifically, grounded in the theoretical insight that full alignment corresponds to a rank-1 Gram matrix, PMRL optimizes the dominant singular value of the representation matrix to align modalities along a shared leading direction. We propose a softmax-based loss function that treats singular values as logits to prioritize the largest singular value. Besides, instance-wise contrastive regularization on the leading eigenvectors maintains inter-instance separability and prevents representation collapse. Extensive experiments across diverse tasks demonstrate PMRL's superiority compared to baseline methods. The source code will be publicly available.

---

## MaskedCLIP: Bridging the Masked and CLIP Space for Semi-Supervised Medical Vision-Language Pre-training

**Link:** [https://arxiv.org/pdf/2507.17239](https://arxiv.org/pdf/2507.17239)

**Similarity:** 0.2358

### Abstract

Foundation models have recently gained tremendous popularity in medical image analysis. State-of-the-art methods leverage either paired image-text data via vision-language pre-training or unpaired image data via self-supervised pre-training to learn foundation models with generalizable image features to boost downstream task performance. However, learning foundation models exclusively on either paired or unpaired image data limits their ability to learn richer and more comprehensive image features. In this paper, we investigate a novel task termed semi-supervised vision-language pre-training, aiming to fully harness the potential of both paired and unpaired image data for foundation model learning. To this end, we propose MaskedCLIP, a synergistic masked image modeling and contrastive language-image pre-training framework for semi-supervised vision-language pre-training. The key challenge in combining paired and unpaired image data for learning a foundation model lies in the incompatible feature spaces derived from these two types of data. To address this issue, we propose to connect the masked feature space with the CLIP feature space with a bridge transformer. In this way, the more semantic specific CLIP features can benefit from the more general masked features for semantic feature extraction. We further propose a masked knowledge distillation loss to distill semantic knowledge of original image features in CLIP feature space back to the predicted masked image features in masked feature space. With this mutually interactive design, our framework effectively leverages both paired and unpaired image data to learn more generalizable image features for downstream tasks. Extensive experiments on retinal image analysis demonstrate the effectiveness and data efficiency of our method.

---

## Toward Scalable Video Narration: A Training-free Approach Using Multimodal Large Language Models

**Link:** [https://arxiv.org/pdf/2507.17050](https://arxiv.org/pdf/2507.17050)

**Similarity:** 0.2318

### Abstract

In this paper, we introduce VideoNarrator, a novel training-free pipeline designed to generate dense video captions that offer a structured snapshot of video content. These captions offer detailed narrations with precise timestamps, capturing the nuances present in each segment of the video. Despite advancements in multimodal large language models (MLLMs) for video comprehension, these models often struggle with temporally aligned narrations and tend to hallucinate, particularly in unfamiliar scenarios. VideoNarrator addresses these challenges by leveraging a flexible pipeline where off-the-shelf MLLMs and visual-language models (VLMs) can function as caption generators, context providers, or caption verifiers. Our experimental results demonstrate that the synergistic interaction of these components significantly enhances the quality and accuracy of video narrations, effectively reducing hallucinations and improving temporal alignment. This structured approach not only enhances video understanding but also facilitates downstream tasks such as video summarization and video question answering, and can be potentially extended for advertising and marketing applications.

---

## The Early Bird Identifies the Worm: You Can't Beat a Head Start in Long-Term Body Re-ID (ECHO-BID)

**Link:** [https://arxiv.org/pdf/2507.17640](https://arxiv.org/pdf/2507.17640)

**Similarity:** 0.2286

### Abstract

Person identification in unconstrained viewing environments presents significant challenges due to variations in distance, viewpoint, imaging conditions, and clothing. We introduce $\textbf{E}$va $\textbf{C}$lothes-Change from $\textbf{H}$idden $\textbf{O}$bjects - $\textbf{B}$ody $\textbf{ID}$entification (ECHO-BID), a class of long-term re-id models built on object-pretrained EVA-02 Large backbones. We compare ECHO-BID to 9 other models that vary systematically in backbone architecture, model size, scale of object classification pretraining, and transfer learning protocol. Models were evaluated on benchmark datasets across constrained, unconstrained, and occluded settings. ECHO-BID, with transfer learning on the most challenging clothes-change data, achieved state-of-the-art results on long-term re-id -- substantially outperforming other methods. ECHO-BID also surpassed other methods by a wide margin in occluded viewing scenarios. A combination of increased model size and Masked Image Modeling during pretraining underlie ECHO-BID's strong performance on long-term re-id. Notably, a smaller, but more challenging transfer learning dataset, generalized better across datasets than a larger, less challenging one. However, the larger dataset with an additional fine-tuning step proved best on the most difficult data. Selecting the correct pretrained backbone architecture and transfer learning protocols can drive substantial gains in long-term re-id performance.

---

## VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization

**Link:** [https://arxiv.org/pdf/2507.17455](https://arxiv.org/pdf/2507.17455)

**Similarity:** 0.2286

### Abstract

Geo-localization from a single image at planet scale (essentially an advanced or extreme version of the kidnapped robot problem) is a fundamental and challenging task in applications such as navigation, autonomous driving and disaster response due to the vast diversity of locations, environmental conditions, and scene variations. Traditional retrieval-based methods for geo-localization struggle with scalability and perceptual aliasing, while classification-based approaches lack generalization and require extensive training data. Recent advances in vision-language models (VLMs) offer a promising alternative by leveraging contextual understanding and reasoning. However, while VLMs achieve high accuracy, they are often prone to hallucinations and lack interpretability, making them unreliable as standalone solutions. In this work, we propose a novel hybrid geo-localization framework that combines the strengths of VLMs with retrieval-based visual place recognition (VPR) methods. Our approach first leverages a VLM to generate a prior, effectively guiding and constraining the retrieval search space. We then employ a retrieval step, followed by a re-ranking mechanism that selects the most geographically plausible matches based on feature similarity and proximity to the initially estimated coordinates. We evaluate our approach on multiple geo-localization benchmarks and show that it consistently outperforms prior state-of-the-art methods, particularly at street (up to 4.51%) and city level (up to 13.52%). Our results demonstrate that VLM-generated geographic priors in combination with VPR lead to scalable, robust, and accurate geo-localization systems.

---

## AURA: A Multi-Modal Medical Agent for Understanding, Reasoning & Annotation

**Link:** [https://arxiv.org/pdf/2507.16940](https://arxiv.org/pdf/2507.16940)

**Similarity:** 0.2276

### Abstract

Recent advancements in Large Language Models (LLMs) have catalyzed a paradigm shift from static prediction systems to agentic AI agents capable of reasoning, interacting with tools, and adapting to complex tasks. While LLM-based agentic systems have shown promise across many domains, their application to medical imaging remains in its infancy. In this work, we introduce AURA, the first visual linguistic explainability agent designed specifically for comprehensive analysis, explanation, and evaluation of medical images. By enabling dynamic interactions, contextual explanations, and hypothesis testing, AURA represents a significant advancement toward more transparent, adaptable, and clinically aligned AI systems. We highlight the promise of agentic AI in transforming medical image analysis from static predictions to interactive decision support. Leveraging Qwen-32B, an LLM-based architecture, AURA integrates a modular toolbox comprising: (i) a segmentation suite with phase grounding, pathology segmentation, and anatomy segmentation to localize clinically meaningful regions; (ii) a counterfactual image-generation module that supports reasoning through image-level explanations; and (iii) a set of evaluation tools including pixel-wise difference-map analysis, classification, and advanced state-of-the-art components to assess diagnostic relevance and visual interpretability.

---

## Weak Links in LinkedIn: Enhancing Fake Profile Detection in the Age of LLMs

**Link:** [https://arxiv.org/pdf/2507.16860](https://arxiv.org/pdf/2507.16860)

**Similarity:** 0.2254

### Abstract

Large Language Models (LLMs) have made it easier to create realistic fake profiles on platforms like LinkedIn. This poses a significant risk for text-based fake profile detectors. In this study, we evaluate the robustness of existing detectors against LLM-generated profiles. While highly effective in detecting manually created fake profiles (False Accept Rate: 6-7%), the existing detectors fail to identify GPT-generated profiles (False Accept Rate: 42-52%). We propose GPT-assisted adversarial training as a countermeasure, restoring the False Accept Rate to between 1-7% without impacting the False Reject Rates (0.5-2%). Ablation studies revealed that detectors trained on combined numerical and textual embeddings exhibit the highest robustness, followed by those using numerical-only embeddings, and lastly those using textual-only embeddings. Complementary analysis on the ability of prompt-based GPT-4Turbo and human evaluators affirms the need for robust automated detectors such as the one proposed in this study.

---

## Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection

**Link:** [https://arxiv.org/pdf/2507.17436](https://arxiv.org/pdf/2507.17436)

**Similarity:** 0.2244

### Abstract

The Mixture of Experts (MoE) architecture has excelled in Large Vision-Language Models (LVLMs), yet its potential in real-time open-vocabulary object detectors, which also leverage large-scale vision-language datasets but smaller models, remains unexplored. This work investigates this domain, revealing intriguing insights. In the shallow layers, experts tend to cooperate with diverse peers to expand the search space. While in the deeper layers, fixed collaborative structures emerge, where each expert maintains 2-3 fixed partners and distinct expert combinations are specialized in processing specific patterns. Concretely, we propose Dynamic-DINO, which extends Grounding DINO 1.5 Edge from a dense model to a dynamic inference framework via an efficient MoE-Tuning strategy. Additionally, we design a granularity decomposition mechanism to decompose the Feed-Forward Network (FFN) of base model into multiple smaller expert networks, expanding the subnet search space. To prevent performance degradation at the start of fine-tuning, we further propose a pre-trained weight allocation strategy for the experts, coupled with a specific router initialization. During inference, only the input-relevant experts are activated to form a compact subnet. Experiments show that, pretrained with merely 1.56M open-source data, Dynamic-DINO outperforms Grounding DINO 1.5 Edge, pretrained on the private Grounding20M dataset.

---

## Harmonization in Magnetic Resonance Imaging: A Survey of Acquisition, Image-level, and Feature-level Methods

**Link:** [https://arxiv.org/pdf/2507.16962](https://arxiv.org/pdf/2507.16962)

**Similarity:** 0.2239

### Abstract

Modern medical imaging technologies have greatly advanced neuroscience research and clinical diagnostics. However, imaging data collected across different scanners, acquisition protocols, or imaging sites often exhibit substantial heterogeneity, known as "batch effects" or "site effects". These non-biological sources of variability can obscure true biological signals, reduce reproducibility and statistical power, and severely impair the generalizability of learning-based models across datasets. Image harmonization aims to eliminate or mitigate such site-related biases while preserving meaningful biological information, thereby improving data comparability and consistency. This review provides a comprehensive overview of key concepts, methodological advances, publicly available datasets, current challenges, and future directions in the field of medical image harmonization, with a focus on magnetic resonance imaging (MRI). We systematically cover the full imaging pipeline, and categorize harmonization approaches into prospective acquisition and reconstruction strategies, retrospective image-level and feature-level methods, and traveling-subject-based techniques. Rather than providing an exhaustive survey, we focus on representative methods, with particular emphasis on deep learning-based approaches. Finally, we summarize the major challenges that remain and outline promising avenues for future research.

---

## A Versatile Pathology Co-pilot via Reasoning Enhanced Multimodal Large Language Model

**Link:** [https://arxiv.org/pdf/2507.17303](https://arxiv.org/pdf/2507.17303)

**Similarity:** 0.2229

### Abstract

Multimodal large language models (MLLMs) have emerged as powerful tools for computational pathology, offering unprecedented opportunities to integrate pathological images with language context for comprehensive diagnostic analysis. These models hold particular promise for automating complex tasks that traditionally require expert interpretation of pathologists. However, current MLLM approaches in pathology demonstrate significantly constrained reasoning capabilities, primarily due to their reliance on expensive chain-of-thought annotations. Additionally, existing methods remain limited to simplex application of visual question answering (VQA) at region-of-interest (ROI) level, failing to address the full spectrum of diagnostic needs such as ROI classification, detection, segmentation, whole-slide-image (WSI) classification and VQA in clinical practice. In this study, we present SmartPath-R1, a versatile MLLM capable of simultaneously addressing both ROI-level and WSI-level tasks while demonstrating robust pathological reasoning capability. Our framework combines scale-dependent supervised fine-tuning and task-aware reinforcement fine-tuning, which circumvents the requirement for chain-of-thought supervision by leveraging the intrinsic knowledge within MLLM. Furthermore, SmartPath-R1 integrates multiscale and multitask analysis through a mixture-of-experts mechanism, enabling dynamic processing for diverse tasks. We curate a large-scale dataset comprising 2.3M ROI samples and 188K WSI samples for training and evaluation. Extensive experiments across 72 tasks validate the effectiveness and superiority of the proposed approach. This work represents a significant step toward developing versatile, reasoning-enhanced AI systems for precision pathology.

---

## Physics-based Human Pose Estimation from a Single Moving RGB Camera

**Link:** [https://arxiv.org/pdf/2507.17406](https://arxiv.org/pdf/2507.17406)

**Similarity:** 0.2211

### Abstract

Most monocular and physics-based human pose tracking methods, while achieving state-of-the-art results, suffer from artifacts when the scene does not have a strictly flat ground plane or when the camera is moving. Moreover, these methods are often evaluated on in-the-wild real world videos without ground-truth data or on synthetic datasets, which fail to model the real world light transport, camera motion, and pose-induced appearance and geometry changes. To tackle these two problems, we introduce MoviCam, the first non-synthetic dataset containing ground-truth camera trajectories of a dynamically moving monocular RGB camera, scene geometry, and 3D human motion with human-scene contact labels. Additionally, we propose PhysDynPose, a physics-based method that incorporates scene geometry and physical constraints for more accurate human motion tracking in case of camera motion and non-flat scenes. More precisely, we use a state-of-the-art kinematics estimator to obtain the human pose and a robust SLAM method to capture the dynamic camera trajectory, enabling the recovery of the human pose in the world frame. We then refine the kinematic pose estimate using our scene-aware physics optimizer. From our new benchmark, we found that even state-of-the-art methods struggle with this inherently challenging setting, i.e. a moving camera and non-planar environments, while our method robustly estimates both human and camera poses in world coordinates.

---

## VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings

**Link:** [https://arxiv.org/pdf/2507.17080](https://arxiv.org/pdf/2507.17080)

**Similarity:** 0.2200

### Abstract

Multimodal learning plays a critical role in e-commerce recommendation platforms today, enabling accurate recommendations and product understanding. However, existing vision-language models, such as CLIP, face key challenges in e-commerce recommendation systems: 1) Weak object-level alignment, where global image embeddings fail to capture fine-grained product attributes, leading to suboptimal retrieval performance; 2) Ambiguous textual representations, where product descriptions often lack contextual clarity, affecting cross-modal matching; and 3) Domain mismatch, as generic vision-language models may not generalize well to e-commerce-specific data. To address these limitations, we propose a framework, VL-CLIP, that enhances CLIP embeddings by integrating Visual Grounding for fine-grained visual understanding and an LLM-based agent for generating enriched text embeddings. Visual Grounding refines image representations by localizing key products, while the LLM agent enhances textual features by disambiguating product descriptions. Our approach significantly improves retrieval accuracy, multimodal retrieval effectiveness, and recommendation quality across tens of millions of items on one of the largest e-commerce platforms in the U.S., increasing CTR by 18.6%, ATC by 15.5%, and GMV by 4.0%. Additional experimental results show that our framework outperforms vision-language models, including CLIP, FashionCLIP, and GCL, in both precision and semantic alignment, demonstrating the potential of combining object-aware visual grounding and LLM-enhanced text representation for robust multimodal recommendations.

---

## InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation

**Link:** [https://arxiv.org/pdf/2507.17520](https://arxiv.org/pdf/2507.17520)

**Similarity:** 0.2192

### Abstract

To operate effectively in the real world, robots must integrate multimodal reasoning with precise action generation. However, existing vision-language-action (VLA) models often sacrifice one for the other, narrow their abilities to task-specific manipulation data, and suffer catastrophic forgetting of pre-trained vision-language capabilities. To bridge this gap, we introduce InstructVLA, an end-to-end VLA model that preserves the flexible reasoning of large vision-language models (VLMs) while delivering leading manipulation performance. InstructVLA introduces a novel training paradigm, Vision-Language-Action Instruction Tuning (VLA-IT), which employs multimodal training with mixture-of-experts adaptation to jointly optimize textual reasoning and action generation on both standard VLM corpora and a curated 650K-sample VLA-IT dataset. On in-domain SimplerEnv tasks, InstructVLA achieves 30.5% improvement over SpatialVLA. To evaluate generalization, we introduce SimplerEnv-Instruct, an 80-task benchmark requiring closed-loop control and high-level instruction understanding, where it outperforms a fine-tuned OpenVLA by 92% and an action expert aided by GPT-4o by 29%. Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings. These results demonstrate InstructVLA's potential for bridging intuitive and steerable human-robot interaction with efficient policy learning.

---

## A Comprehensive Evaluation Framework for the Study of the Effects of Facial Filters on Face Recognition Accuracy

**Link:** [https://arxiv.org/pdf/2507.17729](https://arxiv.org/pdf/2507.17729)

**Similarity:** 0.2175

### Abstract

Facial filters are now commonplace for social media users around the world. Previous work has demonstrated that facial filters can negatively impact automated face recognition performance. However, these studies focus on small numbers of hand-picked filters in particular styles. In order to more effectively incorporate the wide ranges of filters present on various social media applications, we introduce a framework that allows for larger-scale study of the impact of facial filters on automated recognition. This framework includes a controlled dataset of face images, a principled filter selection process that selects a representative range of filters for experimentation, and a set of experiments to evaluate the filters' impact on recognition. We demonstrate our framework with a case study of filters from the American applications Instagram and Snapchat and the Chinese applications Meitu and Pitu to uncover cross-cultural differences. Finally, we show how the filtering effect in a face embedding space can easily be detected and restored to improve face recognition performance.

---

## Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention

**Link:** [https://arxiv.org/pdf/2507.17745](https://arxiv.org/pdf/2507.17745)

**Similarity:** 0.2155

### Abstract

Recent advances in sparse voxel representations have significantly improved the quality of 3D content generation, enabling high-resolution modeling with fine-grained geometry. However, existing frameworks suffer from severe computational inefficiencies due to the quadratic complexity of attention mechanisms in their two-stage diffusion pipelines. In this work, we propose Ultra3D, an efficient 3D generation framework that significantly accelerates sparse voxel modeling without compromising quality. Our method leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, reducing token count and accelerating voxel coordinate prediction. To refine per-voxel latent features in the second stage, we introduce Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions. This design preserves structural continuity while avoiding unnecessary global attention, achieving up to 6.7x speed-up in latent generation. To support this mechanism, we construct a scalable part annotation pipeline that converts raw meshes into part-labeled sparse voxels. Extensive experiments demonstrate that Ultra3D supports high-resolution 3D generation at 1024 resolution and achieves state-of-the-art performance in both visual fidelity and user preference.

---

## Content-based 3D Image Retrieval and a ColBERT-inspired Re-ranking for Tumor Flagging and Staging

**Link:** [https://arxiv.org/pdf/2507.17412](https://arxiv.org/pdf/2507.17412)

**Similarity:** 0.2152

### Abstract

The increasing volume of medical images poses challenges for radiologists in retrieving relevant cases. Content-based image retrieval (CBIR) systems offer potential for efficient access to similar cases, yet lack standardized evaluation and comprehensive studies. Building on prior studies for tumor characterization via CBIR, this study advances CBIR research for volumetric medical images through three key contributions: (1) a framework eliminating reliance on pre-segmented data and organ-specific datasets, aligning with large and unstructured image archiving systems, i.e. PACS in clinical practice; (2) introduction of C-MIR, a novel volumetric re-ranking method adapting ColBERT's contextualized late interaction mechanism for 3D medical imaging; (3) comprehensive evaluation across four tumor sites using three feature extractors and three database configurations. Our evaluations highlight the significant advantages of C-MIR. We demonstrate the successful adaptation of the late interaction principle to volumetric medical images, enabling effective context-aware re-ranking. A key finding is C-MIR's ability to effectively localize the region of interest, eliminating the need for pre-segmentation of datasets and offering a computationally efficient alternative to systems relying on expensive data enrichment steps. C-MIR demonstrates promising improvements in tumor flagging, achieving improved performance, particularly for colon and lung tumors (p<0.05). C-MIR also shows potential for improving tumor staging, warranting further exploration of its capabilities. Ultimately, our work seeks to bridge the gap between advanced retrieval techniques and their practical applications in healthcare, paving the way for improved diagnostic processes.

---

## Monocular Semantic Scene Completion via Masked Recurrent Networks

**Link:** [https://arxiv.org/pdf/2507.17661](https://arxiv.org/pdf/2507.17661)

**Similarity:** 0.2147

### Abstract

Monocular Semantic Scene Completion (MSSC) aims to predict the voxel-wise occupancy and semantic category from a single-view RGB image. Existing methods adopt a single-stage framework that aims to simultaneously achieve visible region segmentation and occluded region hallucination, while also being affected by inaccurate depth estimation. Such methods often achieve suboptimal performance, especially in complex scenes. We propose a novel two-stage framework that decomposes MSSC into coarse MSSC followed by the Masked Recurrent Network. Specifically, we propose the Masked Sparse Gated Recurrent Unit (MS-GRU) which concentrates on the occupied regions by the proposed mask updating mechanism, and a sparse GRU design is proposed to reduce the computation cost. Additionally, we propose the distance attention projection to reduce projection errors by assigning different attention scores according to the distance to the observed surface. Experimental results demonstrate that our proposed unified framework, MonoMRN, effectively supports both indoor and outdoor scenes and achieves state-of-the-art performance on the NYUv2 and SemanticKITTI datasets. Furthermore, we conduct robustness analysis under various disturbances, highlighting the role of the Masked Recurrent Network in enhancing the model's resilience to such challenges. The source code is publicly available.

---

## EndoGen: Conditional Autoregressive Endoscopic Video Generation

**Link:** [https://arxiv.org/pdf/2507.17388](https://arxiv.org/pdf/2507.17388)

**Similarity:** 0.2131

### Abstract

Endoscopic video generation is crucial for advancing medical imaging and enhancing diagnostic capabilities. However, prior efforts in this field have either focused on static images, lacking the dynamic context required for practical applications, or have relied on unconditional generation that fails to provide meaningful references for clinicians. Therefore, in this paper, we propose the first conditional endoscopic video generation framework, namely EndoGen. Specifically, we build an autoregressive model with a tailored Spatiotemporal Grid-Frame Patterning (SGP) strategy. It reformulates the learning of generating multiple frames as a grid-based image generation pattern, which effectively capitalizes the inherent global dependency modeling capabilities of autoregressive architectures. Furthermore, we propose a Semantic-Aware Token Masking (SAT) mechanism, which enhances the model's ability to produce rich and diverse content by selectively focusing on semantically meaningful regions during the generation process. Through extensive experiments, we demonstrate the effectiveness of our framework in generating high-quality, conditionally guided endoscopic content, and improves the performance of downstream task of polyp segmentation. Code released at this https URL.

---

## Coarse-to-fine crack cue for robust crack detection

**Link:** [https://arxiv.org/pdf/2507.16851](https://arxiv.org/pdf/2507.16851)

**Similarity:** 0.2126

### Abstract

Crack detection is an important task in computer vision. Despite impressive in-dataset performance, deep learning-based methods still struggle in generalizing to unseen domains. The thin structure property of cracks is usually overlooked by previous methods. In this work, we introduce CrackCue, a novel method for robust crack detection based on coarse-to-fine crack cue generation. The core concept lies on leveraging the thin structure property to generate a robust crack cue, guiding the crack detection. Specifically, we first employ a simple max-pooling and upsampling operation on the crack image. This results in a coarse crack-free background, based on which a fine crack-free background can be obtained via a reconstruction network. The difference between the original image and fine crack-free background provides a fine crack cue. This fine cue embeds robust crack prior information which is unaffected by complex backgrounds, shadow, and varied lighting. As a plug-and-play method, we incorporate the proposed CrackCue into three advanced crack detection networks. Extensive experimental results demonstrate that the proposed CrackCue significantly improves the generalization ability and robustness of the baseline methods. The source code will be publicly available.

---

## HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs

**Link:** [https://arxiv.org/pdf/2507.17394](https://arxiv.org/pdf/2507.17394)

**Similarity:** 0.2122

### Abstract

Video Anomaly Detection (VAD) aims to identify and locate deviations from normal patterns in video sequences. Traditional methods often struggle with substantial computational demands and a reliance on extensive labeled datasets, thereby restricting their practical applicability. To address these constraints, we propose HiProbe-VAD, a novel framework that leverages pre-trained Multimodal Large Language Models (MLLMs) for VAD without requiring fine-tuning. In this paper, we discover that the intermediate hidden states of MLLMs contain information-rich representations, exhibiting higher sensitivity and linear separability for anomalies compared to the output layer. To capitalize on this, we propose a Dynamic Layer Saliency Probing (DLSP) mechanism that intelligently identifies and extracts the most informative hidden states from the optimal intermediate layer during the MLLMs reasoning. Then a lightweight anomaly scorer and temporal localization module efficiently detects anomalies using these extracted hidden states and finally generate explanations. Experiments on the UCF-Crime and XD-Violence datasets demonstrate that HiProbe-VAD outperforms existing training-free and most traditional approaches. Furthermore, our framework exhibits remarkable cross-model generalization capabilities in different MLLMs without any tuning, unlocking the potential of pre-trained MLLMs for video anomaly detection and paving the way for more practical and scalable solutions.

---

## Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras

**Link:** [https://arxiv.org/pdf/2507.17664](https://arxiv.org/pdf/2507.17664)

**Similarity:** 0.2095

### Abstract

Event cameras offer microsecond-level latency and robustness to motion blur, making them ideal for understanding dynamic environments. Yet, connecting these asynchronous streams to human language remains an open challenge. We introduce Talk2Event, the first large-scale benchmark for language-driven object grounding in event-based perception. Built from real-world driving data, we provide over 30,000 validated referring expressions, each enriched with four grounding attributes -- appearance, status, relation to viewer, and relation to other objects -- bridging spatial, temporal, and relational reasoning. To fully exploit these cues, we propose EventRefer, an attribute-aware grounding framework that dynamically fuses multi-attribute representations through a Mixture of Event-Attribute Experts (MoEE). Our method adapts to different modalities and scene dynamics, achieving consistent gains over state-of-the-art baselines in event-only, frame-only, and event-frame fusion settings. We hope our dataset and approach will establish a foundation for advancing multimodal, temporally-aware, and language-driven perception in real-world robotics and autonomy.

---

## CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits

**Link:** [https://arxiv.org/pdf/2507.17327](https://arxiv.org/pdf/2507.17327)

**Similarity:** 0.2090

### Abstract

With the rapid advancement of large foundation models, AIGC, cloud rendering, and real-time motion capture technologies, digital humans are now capable of achieving synchronized facial expressions and body movements, engaging in intelligent dialogues driven by natural language, and enabling the fast creation of personalized avatars. While current mainstream approaches to digital humans primarily focus on 3D models and 2D video-based representations, interactive 2D cartoon-style digital humans have received relatively less attention. Compared to 3D digital humans that require complex modeling and high rendering costs, and 2D video-based solutions that lack flexibility and real-time interactivity, 2D cartoon-style Live2D models offer a more efficient and expressive alternative. By simulating 3D-like motion through layered segmentation without the need for traditional 3D modeling, Live2D enables dynamic and real-time manipulation. In this technical report, we present CartoonAlive, an innovative method for generating high-quality Live2D digital humans from a single input portrait image. CartoonAlive leverages the shape basis concept commonly used in 3D face modeling to construct facial blendshapes suitable for Live2D. It then infers the corresponding blendshape weights based on facial keypoints detected from the input image. This approach allows for the rapid generation of a highly expressive and visually accurate Live2D model that closely resembles the input portrait, within less than half a minute. Our work provides a practical and scalable solution for creating interactive 2D cartoon characters, opening new possibilities in digital content creation and virtual character animation. The project homepage is this https URL.

---

## HLFormer: Enhancing Partially Relevant Video Retrieval with Hyperbolic Learning

**Link:** [https://arxiv.org/pdf/2507.17402](https://arxiv.org/pdf/2507.17402)

**Similarity:** 0.2085

### Abstract

Partially Relevant Video Retrieval (PRVR) addresses the critical challenge of matching untrimmed videos with text queries describing only partial content. Existing methods suffer from geometric distortion in Euclidean space that sometimes misrepresents the intrinsic hierarchical structure of videos and overlooks certain hierarchical semantics, ultimately leading to suboptimal temporal modeling. To address this issue, we propose the first hyperbolic modeling framework for PRVR, namely HLFormer, which leverages hyperbolic space learning to compensate for the suboptimal hierarchical modeling capabilities of Euclidean space. Specifically, HLFormer integrates the Lorentz Attention Block and Euclidean Attention Block to encode video embeddings in hybrid spaces, using the Mean-Guided Adaptive Interaction Module to dynamically fuse features. Additionally, we introduce a Partial Order Preservation Loss to enforce "text < video" hierarchy through Lorentzian cone constraints. This approach further enhances cross-modal matching by reinforcing partial relevance between video content and text queries. Extensive experiments show that HLFormer outperforms state-of-the-art methods. Code is released at this https URL.

---

## CAPRI-CT: Causal Analysis and Predictive Reasoning for Image Quality Optimization in Computed Tomography

**Link:** [https://arxiv.org/pdf/2507.17420](https://arxiv.org/pdf/2507.17420)

**Similarity:** 0.2079

### Abstract

In computed tomography (CT), achieving high image quality while minimizing radiation exposure remains a key clinical challenge. This paper presents CAPRI-CT, a novel causal-aware deep learning framework for Causal Analysis and Predictive Reasoning for Image Quality Optimization in CT imaging. CAPRI-CT integrates image data with acquisition metadata (such as tube voltage, tube current, and contrast agent types) to model the underlying causal relationships that influence image quality. An ensemble of Variational Autoencoders (VAEs) is employed to extract meaningful features and generate causal representations from observational data, including CT images and associated imaging parameters. These input features are fused to predict the Signal-to-Noise Ratio (SNR) and support counterfactual inference, enabling what-if simulations, such as changes in contrast agents (types and concentrations) or scan parameters. CAPRI-CT is trained and validated using an ensemble learning approach, achieving strong predictive performance. By facilitating both prediction and interpretability, CAPRI-CT provides actionable insights that could help radiologists and technicians design more efficient CT protocols without repeated physical scans. The source code and dataset are publicly available at this https URL.

---

## CLAMP: Contrastive Learning with Adaptive Multi-loss and Progressive Fusion for Multimodal Aspect-Based Sentiment Analysis

**Link:** [https://arxiv.org/pdf/2507.16854](https://arxiv.org/pdf/2507.16854)

**Similarity:** 0.2055

### Abstract

Multimodal aspect-based sentiment analysis(MABSA) seeks to identify aspect terms within paired image-text data and determine their fine grained sentiment polarities, representing a fundamental task for improving the effectiveness of applications such as product review systems and public opinion monitoring. Existing methods face challenges such as cross modal alignment noise and insufficient consistency in fine-grained representations. While global modality alignment methods often overlook the connection between aspect terms and their corresponding local visual regions, bridging the representation gap between text and images remains a challenge. To address these limitations, this paper introduces an end to end Contrastive Learning framework with Adaptive Multi-loss and Progressive Attention Fusion(CLAMP). The framework is composed of three novel modules: Progressive Attention Fusion network, Multi-task Contrastive Learning, and Adaptive Multi-loss Aggregation. The Progressive Attention Fusion network enhances fine-grained alignment between textual features and image regions via hierarchical, multi-stage cross modal interactions, effectively suppressing irrelevant visual noise. Secondly, multi-task contrastive learning combines global modal contrast and local granularity alignment to enhance cross modal representation consistency. Adaptive Multi-loss Aggregation employs a dynamic uncertainty based weighting mechanism to calibrate loss contributions according to each task's uncertainty, thereby mitigating gradient interference. Evaluation on standard public benchmarks demonstrates that CLAMP consistently outperforms the vast majority of existing state of the art methods.

---

## MCM: Mamba-based Cardiac Motion Tracking using Sequential Images in MRI

**Link:** [https://arxiv.org/pdf/2507.17678](https://arxiv.org/pdf/2507.17678)

**Similarity:** 0.2040

### Abstract

Myocardial motion tracking is important for assessing cardiac function and diagnosing cardiovascular diseases, for which cine cardiac magnetic resonance (CMR) has been established as the gold standard imaging modality. Many existing methods learn motion from single image pairs consisting of a reference frame and a randomly selected target frame from the cardiac cycle. However, these methods overlook the continuous nature of cardiac motion and often yield inconsistent and non-smooth motion estimations. In this work, we propose a novel Mamba-based cardiac motion tracking network (MCM) that explicitly incorporates target image sequence from the cardiac cycle to achieve smooth and temporally consistent motion tracking. By developing a bi-directional Mamba block equipped with a bi-directional scanning mechanism, our method facilitates the estimation of plausible deformation fields. With our proposed motion decoder that integrates motion information from frames adjacent to the target frame, our method further enhances temporal coherence. Moreover, by taking advantage of Mamba's structured state-space formulation, the proposed method learns the continuous dynamics of the myocardium from sequential images without increasing computational complexity. We evaluate the proposed method on two public datasets. The experimental results demonstrate that the proposed method quantitatively and qualitatively outperforms both conventional and state-of-the-art learning-based cardiac motion tracking methods. The code is available at this https URL.

---

## Reusing Attention for One-stage Lane Topology Understanding

**Link:** [https://arxiv.org/pdf/2507.17617](https://arxiv.org/pdf/2507.17617)

**Similarity:** 0.2032

### Abstract

Understanding lane toplogy relationships accurately is critical for safe autonomous driving. However, existing two-stage methods suffer from inefficiencies due to error propagations and increased computational overheads. To address these challenges, we propose a one-stage architecture that simultaneously predicts traffic elements, lane centerlines and topology relationship, improving both the accuracy and inference speed of lane topology understanding for autonomous driving. Our key innovation lies in reusing intermediate attention resources within distinct transformer decoders. This approach effectively leverages the inherent relational knowledge within the element detection module to enable the modeling of topology relationships among traffic elements and lanes without requiring additional computationally expensive graph networks. Furthermore, we are the first to demonstrate that knowledge can be distilled from models that utilize standard definition (SD) maps to those operates without using SD maps, enabling superior performance even in the absence of SD maps. Extensive experiments on the OpenLane-V2 dataset show that our approach outperforms baseline methods in both accuracy and efficiency, achieving superior results in lane detection, traffic element identification, and topology reasoning. Our code is available at this https URL.

---

## Bringing Balance to Hand Shape Classification: Mitigating Data Imbalance Through Generative Models

**Link:** [https://arxiv.org/pdf/2507.17008](https://arxiv.org/pdf/2507.17008)

**Similarity:** 0.2030

### Abstract

Most sign language handshape datasets are severely limited and unbalanced, posing significant challenges to effective model training. In this paper, we explore the effectiveness of augmenting the training data of a handshape classifier by generating synthetic data. We use an EfficientNet classifier trained on the RWTH German sign language handshape dataset, which is small and heavily unbalanced, applying different strategies to combine generated and real images. We compare two Generative Adversarial Networks (GAN) architectures for data generation: ReACGAN, which uses label information to condition the data generation process through an auxiliary classifier, and SPADE, which utilizes spatially-adaptive normalization to condition the generation on pose information. ReACGAN allows for the generation of realistic images that align with specific handshape labels, while SPADE focuses on generating images with accurate spatial handshape configurations. Our proposed techniques improve the current state-of-the-art accuracy on the RWTH dataset by 5%, addressing the limitations of small and unbalanced datasets. Additionally, our method demonstrates the capability to generalize across different sign language datasets by leveraging pose-based generation trained on the extensive HaGRID dataset. We achieve comparable performance to single-source trained classifiers without the need for retraining the generator.

---

## Assessing Medical Training Skills via Eye and Head Movements

**Link:** [https://arxiv.org/pdf/2507.16819](https://arxiv.org/pdf/2507.16819)

**Similarity:** 0.2020

### Abstract

We examined eye and head movements to gain insights into skill development in clinical settings. A total of 24 practitioners participated in simulated baby delivery training sessions. We calculated key metrics, including pupillary response rate, fixation duration, or angular velocity. Our findings indicate that eye and head tracking can effectively differentiate between trained and untrained practitioners, particularly during labor tasks. For example, head-related features achieved an F1 score of 0.85 and AUC of 0.86, whereas pupil-related features achieved F1 score of 0.77 and AUC of 0.85. The results lay the groundwork for computational models that support implicit skill assessment and training in clinical settings by using commodity eye-tracking glasses as a complementary device to more traditional evaluation methods such as subjective scores.

---

## IONext: Unlocking the Next Era of Inertial Odometry

**Link:** [https://arxiv.org/pdf/2507.17089](https://arxiv.org/pdf/2507.17089)

**Similarity:** 0.2003

### Abstract

Researchers have increasingly adopted Transformer-based models for inertial odometry. While Transformers excel at modeling long-range dependencies, their limited sensitivity to local, fine-grained motion variations and lack of inherent inductive biases often hinder localization accuracy and generalization. Recent studies have shown that incorporating large-kernel convolutions and Transformer-inspired architectural designs into CNN can effectively expand the receptive field, thereby improving global motion perception. Motivated by these insights, we propose a novel CNN-based module called the Dual-wing Adaptive Dynamic Mixer (DADM), which adaptively captures both global motion patterns and local, fine-grained motion features from dynamic inputs. This module dynamically generates selective weights based on the input, enabling efficient multi-scale feature aggregation. To further improve temporal modeling, we introduce the Spatio-Temporal Gating Unit (STGU), which selectively extracts representative and task-relevant motion features in the temporal domain. This unit addresses the limitations of temporal modeling observed in existing CNN approaches. Built upon DADM and STGU, we present a new CNN-based inertial odometry backbone, named Next Era of Inertial Odometry (IONext). Extensive experiments on six public datasets demonstrate that IONext consistently outperforms state-of-the-art (SOTA) Transformer- and CNN-based methods. For instance, on the RNIN dataset, IONext reduces the average ATE by 10% and the average RTE by 12% compared to the representative model iMOT.

---

## From Scan to Action: Leveraging Realistic Scans for Embodied Scene Understanding

**Link:** [https://arxiv.org/pdf/2507.17585](https://arxiv.org/pdf/2507.17585)

**Similarity:** 0.1986

### Abstract

Real-world 3D scene-level scans offer realism and can enable better real-world generalizability for downstream applications. However, challenges such as data volume, diverse annotation formats, and tool compatibility limit their use. This paper demonstrates a methodology to effectively leverage these scans and their annotations. We propose a unified annotation integration using USD, with application-specific USD flavors. We identify challenges in utilizing holistic real-world scan datasets and present mitigation strategies. The efficacy of our approach is demonstrated through two downstream applications: LLM-based scene editing, enabling effective LLM understanding and adaptation of the data (80% success), and robotic simulation, achieving an 87% success rate in policy learning.

---

## DesignLab: Designing Slides Through Iterative Detection and Correction

**Link:** [https://arxiv.org/pdf/2507.17202](https://arxiv.org/pdf/2507.17202)

**Similarity:** 0.1985

### Abstract

Designing high-quality presentation slides can be challenging for non-experts due to the complexity involved in navigating various design choices. Numerous automated tools can suggest layouts and color schemes, yet often lack the ability to refine their own output, which is a key aspect in real-world workflows. We propose DesignLab, which separates the design process into two roles, the design reviewer, who identifies design-related issues, and the design contributor who corrects them. This decomposition enables an iterative loop where the reviewer continuously detects issues and the contributor corrects them, allowing a draft to be further polished with each iteration, reaching qualities that were unattainable. We fine-tune large language models for these roles and simulate intermediate drafts by introducing controlled perturbations, enabling the design reviewer learn design errors and the contributor learn how to fix them. Our experiments show that DesignLab outperforms existing design-generation methods, including a commercial tool, by embracing the iterative nature of designing which can result in polished, professional slides.

---

## Controllable Video Generation: A Survey

**Link:** [https://arxiv.org/pdf/2507.16869](https://arxiv.org/pdf/2507.16869)

**Similarity:** 0.1985

### Abstract

With the rapid development of AI-generated content (AIGC), video generation has emerged as one of its most dynamic and impactful subfields. In particular, the advancement of video generation foundation models has led to growing demand for controllable video generation methods that can more accurately reflect user intent. Most existing foundation models are designed for text-to-video generation, where text prompts alone are often insufficient to express complex, multi-modal, and fine-grained user requirements. This limitation makes it challenging for users to generate videos with precise control using current models. To address this issue, recent research has explored the integration of additional non-textual conditions, such as camera motion, depth maps, and human pose, to extend pretrained video generation models and enable more controllable video synthesis. These approaches aim to enhance the flexibility and practical applicability of AIGC-driven video generation systems. In this survey, we provide a systematic review of controllable video generation, covering both theoretical foundations and recent advances in the field. We begin by introducing the key concepts and commonly used open-source video generation models. We then focus on control mechanisms in video diffusion models, analyzing how different types of conditions can be incorporated into the denoising process to guide generation. Finally, we categorize existing methods based on the types of control signals they leverage, including single-condition generation, multi-condition generation, and universal controllable generation. For a complete list of the literature on controllable video generation reviewed, please visit our curated repository at this https URL.

---

## FedVLM: Scalable Personalized Vision-Language Models through Federated Learning

**Link:** [https://arxiv.org/pdf/2507.17088](https://arxiv.org/pdf/2507.17088)

**Similarity:** 0.1983

### Abstract

Vision-language models (VLMs) demonstrate impressive zero-shot and few-shot learning capabilities, making them essential for several downstream tasks. However, fine-tuning these models at scale remains challenging, particularly in federated environments where data is decentralized and non-iid across clients. Existing parameter-efficient tuning methods like LoRA (Low-Rank Adaptation) reduce computational overhead but struggle with heterogeneous client data, leading to suboptimal generalization. To address these challenges, we propose FedVLM, a federated LoRA fine-tuning framework that enables decentralized adaptation of VLMs while preserving model privacy and reducing reliance on centralized training. To further tackle data heterogeneity, we introduce personalized LoRA (pLoRA), which dynamically adapts LoRA parameters to each client's unique data distribution, significantly improving local adaptation while maintaining global model aggregation. Experiments on the RLAIF-V dataset show that pLoRA improves client-specific performance by 24.5% over standard LoRA, demonstrating superior adaptation in non-iid settings. FedVLM provides a scalable and efficient solution for fine-tuning VLMs in federated settings, advancing personalized adaptation in distributed learning scenarios.

---

## Dynamic Scoring with Enhanced Semantics for Training-Free Human-Object Interaction Detection

**Link:** [https://arxiv.org/pdf/2507.17456](https://arxiv.org/pdf/2507.17456)

**Similarity:** 0.1973

### Abstract

Human-Object Interaction (HOI) detection aims to identify humans and objects within images and interpret their interactions. Existing HOI methods rely heavily on large datasets with manual annotations to learn interactions from visual cues. These annotations are labor-intensive to create, prone to inconsistency, and limit scalability to new domains and rare interactions. We argue that recent advances in Vision-Language Models (VLMs) offer untapped potential, particularly in enhancing interaction representation. While prior work has injected such potential and even proposed training-free methods, there remain key gaps. Consequently, we propose a novel training-free HOI detection framework for Dynamic Scoring with enhanced semantics (DYSCO) that effectively utilizes textual and visual interaction representations within a multimodal registry, enabling robust and nuanced interaction understanding. This registry incorporates a small set of visual cues and uses innovative interaction signatures to improve the semantic alignment of verbs, facilitating effective generalization to rare interactions. Additionally, we propose a unique multi-head attention mechanism that adaptively weights the contributions of the visual and textual features. Experimental results demonstrate that our DYSCO surpasses training-free state-of-the-art models and is competitive with training-based approaches, particularly excelling in rare interactions. Code is available at this https URL.

---

## Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors

**Link:** [https://arxiv.org/pdf/2507.16850](https://arxiv.org/pdf/2507.16850)

**Similarity:** 0.1953

### Abstract

Monocular 3D human pose estimation remains a challenging and ill-posed problem, particularly in real-time settings and unconstrained environments. While direct imageto-3D approaches require large annotated datasets and heavy models, 2D-to-3D lifting offers a more lightweight and flexible alternative-especially when enhanced with prior knowledge. In this work, we propose a framework that combines real-time 2D keypoint detection with geometry-aware 2D-to-3D lifting, explicitly leveraging known camera intrinsics and subject-specific anatomical priors. Our approach builds on recent advances in self-calibration and biomechanically-constrained inverse kinematics to generate large-scale, plausible 2D-3D training pairs from MoCap and synthetic datasets. We discuss how these ingredients can enable fast, personalized, and accurate 3D pose estimation from monocular images without requiring specialized hardware. This proposal aims to foster discussion on bridging data-driven learning and model-based priors to improve accuracy, interpretability, and deployability of 3D human motion capture on edge devices in the wild.

---

## Divisive Decisions: Improving Salience-Based Training for Generalization in Binary Classification Tasks

**Link:** [https://arxiv.org/pdf/2507.17000](https://arxiv.org/pdf/2507.17000)

**Similarity:** 0.1951

### Abstract

Existing saliency-guided training approaches improve model generalization by incorporating a loss term that compares the model's class activation map (CAM) for a sample's true-class ({\it i.e.}, correct-label class) against a human reference saliency map. However, prior work has ignored the false-class CAM(s), that is the model's saliency obtained for incorrect-label class. We hypothesize that in binary tasks the true and false CAMs should diverge on the important classification features identified by humans (and reflected in human saliency maps). We use this hypothesis to motivate three new saliency-guided training methods incorporating both true- and false-class model's CAM into the training strategy and a novel post-hoc tool for identifying important features. We evaluate all introduced methods on several diverse binary close-set and open-set classification tasks, including synthetic face detection, biometric presentation attack detection, and classification of anomalies in chest X-ray scans, and find that the proposed methods improve generalization capabilities of deep learning models over traditional (true-class CAM only) saliency-guided training approaches. We offer source codes and model weights\footnote{GitHub repository link removed to preserve anonymity} to support reproducible research.

---

## SIA: Enhancing Safety via Intent Awareness for Vision-Language Models

**Link:** [https://arxiv.org/pdf/2507.16856](https://arxiv.org/pdf/2507.16856)

**Similarity:** 0.1942

### Abstract

As vision-language models (VLMs) are increasingly deployed in real-world applications, new safety risks arise from the subtle interplay between images and text. In particular, seemingly innocuous inputs can combine to reveal harmful intent, leading to unsafe model responses. Despite increasing attention to multimodal safety, previous approaches based on post hoc filtering or static refusal prompts struggle to detect such latent risks, especially when harmfulness emerges only from the combination of inputs. We propose SIA (Safety via Intent Awareness), a training-free prompt engineering framework that proactively detects and mitigates harmful intent in multimodal inputs. SIA employs a three-stage reasoning process: (1) visual abstraction via captioning, (2) intent inference through few-shot chain-of-thought prompting, and (3) intent-conditioned response refinement. Rather than relying on predefined rules or classifiers, SIA dynamically adapts to the implicit intent inferred from the image-text pair. Through extensive experiments on safety-critical benchmarks including SIUO, MM-SafetyBench, and HoliSafe, we demonstrate that SIA achieves substantial safety improvements, outperforming prior methods. Although SIA shows a minor reduction in general reasoning accuracy on MMStar, the corresponding safety gains highlight the value of intent-aware reasoning in aligning VLMs with human-centric values.

---

## Automated Hybrid Grounding Using Structural and Data-Driven Heuristics

**Link:** [https://arxiv.org/pdf/2507.17493](https://arxiv.org/pdf/2507.17493)

**Similarity:** 0.1933

### Abstract

The grounding bottleneck poses one of the key challenges that hinders the widespread adoption of Answer Set Programming in industry. Hybrid Grounding is a step in alleviating the bottleneck by combining the strength of standard bottom-up grounding with recently proposed techniques where rule bodies are decoupled during grounding. However, it has remained unclear when hybrid grounding shall use body-decoupled grounding and when to use standard bottom-up grounding. In this paper, we address this issue by developing automated hybrid grounding: we introduce a splitting algorithm based on data-structural heuristics that detects when to use body-decoupled grounding and when standard grounding is beneficial. We base our heuristics on the structure of rules and an estimation procedure that incorporates the data of the instance. The experiments conducted on our prototypical implementation demonstrate promising results, which show an improvement on hard-to-ground scenarios, whereas on hard-to-solve instances we approach state-of-the-art performance.

---

## ReMeREC: Relation-aware and Multi-entity Referring Expression Comprehension

**Link:** [https://arxiv.org/pdf/2507.16877](https://arxiv.org/pdf/2507.16877)

**Similarity:** 0.1932

### Abstract

Referring Expression Comprehension (REC) aims to localize specified entities or regions in an image based on natural language descriptions. While existing methods handle single-entity localization, they often ignore complex inter-entity relationships in multi-entity scenes, limiting their accuracy and reliability. Additionally, the lack of high-quality datasets with fine-grained, paired image-text-relation annotations hinders further progress. To address this challenge, we first construct a relation-aware, multi-entity REC dataset called ReMeX, which includes detailed relationship and textual annotations. We then propose ReMeREC, a novel framework that jointly leverages visual and textual cues to localize multiple entities while modeling their inter-relations. To address the semantic ambiguity caused by implicit entity boundaries in language, we introduce the Text-adaptive Multi-entity Perceptron (TMP), which dynamically infers both the quantity and span of entities from fine-grained textual cues, producing distinctive representations. Additionally, our Entity Inter-relationship Reasoner (EIR) enhances relational reasoning and global scene understanding. To further improve language comprehension for fine-grained prompts, we also construct a small-scale auxiliary dataset, EntityText, generated using large language models. Experiments on four benchmark datasets show that ReMeREC achieves state-of-the-art performance in multi-entity grounding and relation prediction, outperforming existing approaches by a large margin.

---

## CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts

**Link:** [https://arxiv.org/pdf/2507.17651](https://arxiv.org/pdf/2507.17651)

**Similarity:** 0.1922

### Abstract

An important challenge when using computer vision models in the real world is to evaluate their performance in potential out-of-distribution (OOD) scenarios. While simple synthetic corruptions are commonly applied to test OOD robustness, they often fail to capture nuisance shifts that occur in the real world. Recently, diffusion models have been applied to generate realistic images for benchmarking, but they are restricted to binary nuisance shifts. In this work, we introduce CNS-Bench, a Continuous Nuisance Shift Benchmark to quantify OOD robustness of image classifiers for continuous and realistic generative nuisance shifts. CNS-Bench allows generating a wide range of individual nuisance shifts in continuous severities by applying LoRA adapters to diffusion models. To address failure cases, we propose a filtering mechanism that outperforms previous methods, thereby enabling reliable benchmarking with generative models. With the proposed benchmark, we perform a large-scale study to evaluate the robustness of more than 40 classifiers under various nuisance shifts. Through carefully designed comparisons and analyses, we find that model rankings can change for varying shifts and shift scales, which cannot be captured when applying common binary shifts. Additionally, we show that evaluating the model performance on a continuous scale allows the identification of model failure points, providing a more nuanced understanding of model robustness. Project page including code and data: this https URL.

---

## PIG-Nav: Key Insights for Pretrained Image Goal Navigation Models

**Link:** [https://arxiv.org/pdf/2507.17220](https://arxiv.org/pdf/2507.17220)

**Similarity:** 0.1920

### Abstract

Recent studies have explored pretrained (foundation) models for vision-based robotic navigation, aiming to achieve generalizable navigation and positive transfer across diverse environments while enhancing zero-shot performance in unseen settings. In this work, we introduce PIG-Nav (Pretrained Image-Goal Navigation), a new approach that further investigates pretraining strategies for vision-based navigation models and contributes in two key areas. Model-wise, we identify two critical design choices that consistently improve the performance of pretrained navigation models: (1) integrating an early-fusion network structure to combine visual observations and goal images via appropriately pretrained Vision Transformer (ViT) image encoder, and (2) introducing suitable auxiliary tasks to enhance global navigation representation learning, thus further improving navigation performance. Dataset-wise, we propose a novel data preprocessing pipeline for efficiently labeling large-scale game video datasets for navigation model training. We demonstrate that augmenting existing open navigation datasets with diverse gameplay videos improves model performance. Our model achieves an average improvement of 22.6% in zero-shot settings and a 37.5% improvement in fine-tuning settings over existing visual navigation foundation models in two complex simulated environments and one real-world environment. These results advance the state-of-the-art in pretrained image-goal navigation models. Notably, our model maintains competitive performance while requiring significantly less fine-tuning data, highlighting its potential for real-world deployment with minimal labeled supervision.

---

## Yume: An Interactive World Generation Model

**Link:** [https://arxiv.org/pdf/2507.17744](https://arxiv.org/pdf/2507.17744)

**Similarity:** 0.1910

### Abstract

Yume aims to use images, text, or videos to create an interactive, realistic, and dynamic world, which allows exploration and control using peripheral devices or neural signals. In this report, we present a preview version of \method, which creates a dynamic world from an input image and allows exploration of the world using keyboard actions. To achieve this high-fidelity and interactive video world generation, we introduce a well-designed framework, which consists of four main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration. First, we quantize camera motions for stable training and user-friendly interaction using keyboard inputs. Then, we introduce the Masked Video Diffusion Transformer~(MVDT) with a memory module for infinite video generation in an autoregressive manner. After that, training-free Anti-Artifact Mechanism (AAM) and Time Travel Sampling based on Stochastic Differential Equations (TTS-SDE) are introduced to the sampler for better visual quality and more precise control. Moreover, we investigate model acceleration by synergistic optimization of adversarial distillation and caching mechanisms. We use the high-quality world exploration dataset \sekai to train \method, and it achieves remarkable results in diverse scenes and applications. All data, codebase, and model weights are available on this https URL. Yume will update monthly to achieve its original goal. Project page: this https URL.

---

## Sparser2Sparse: Single-shot Sparser-to-Sparse Learning for Spatial Transcriptomics Imputation with Natural Image Co-learning

**Link:** [https://arxiv.org/pdf/2507.16886](https://arxiv.org/pdf/2507.16886)

**Similarity:** 0.1897

### Abstract

Spatial transcriptomics (ST) has revolutionized biomedical research by enabling high resolution gene expression profiling within tissues. However, the high cost and scarcity of high resolution ST data remain significant challenges. We present Single-shot Sparser-to-Sparse (S2S-ST), a novel framework for accurate ST imputation that requires only a single and low-cost sparsely sampled ST dataset alongside widely available natural images for co-training. Our approach integrates three key innovations: (1) a sparser-to-sparse self-supervised learning strategy that leverages intrinsic spatial patterns in ST data, (2) cross-domain co-learning with natural images to enhance feature representation, and (3) a Cascaded Data Consistent Imputation Network (CDCIN) that iteratively refines predictions while preserving sampled gene data fidelity. Extensive experiments on diverse tissue types, including breast cancer, liver, and lymphoid tissue, demonstrate that our method outperforms state-of-the-art approaches in imputation accuracy. By enabling robust ST reconstruction from sparse inputs, our framework significantly reduces reliance on costly high resolution data, facilitating potential broader adoption in biomedical research and clinical applications.

---

## StreamME: Simplify 3D Gaussian Avatar within Live Stream

**Link:** [https://arxiv.org/pdf/2507.17029](https://arxiv.org/pdf/2507.17029)

**Similarity:** 0.1873

### Abstract

We propose StreamME, a method focuses on fast 3D avatar reconstruction. The StreamME synchronously records and reconstructs a head avatar from live video streams without any pre-cached data, enabling seamless integration of the reconstructed appearance into downstream applications. This exceptionally fast training strategy, which we refer to as on-the-fly training, is central to our approach. Our method is built upon 3D Gaussian Splatting (3DGS), eliminating the reliance on MLPs in deformable 3DGS and relying solely on geometry, which significantly improves the adaptation speed to facial expression. To further ensure high efficiency in on-the-fly training, we introduced a simplification strategy based on primary points, which distributes the point clouds more sparsely across the facial surface, optimizing points number while maintaining rendering quality. Leveraging the on-the-fly training capabilities, our method protects the facial privacy and reduces communication bandwidth in VR system or online conference. Additionally, it can be directly applied to downstream application such as animation, toonify, and relighting. Please refer to our project page for more details: this https URL.

---

## Post-Disaster Affected Area Segmentation with a Vision Transformer (ViT)-based EVAP Model using Sentinel-2 and Formosat-5 Imagery

**Link:** [https://arxiv.org/pdf/2507.16849](https://arxiv.org/pdf/2507.16849)

**Similarity:** 0.1851

### Abstract

We propose a vision transformer (ViT)-based deep learning framework to refine disaster-affected area segmentation from remote sensing imagery, aiming to support and enhance the Emergent Value Added Product (EVAP) developed by the Taiwan Space Agency (TASA). The process starts with a small set of manually annotated regions. We then apply principal component analysis (PCA)-based feature space analysis and construct a confidence index (CI) to expand these labels, producing a weakly supervised training set. These expanded labels are then used to train ViT-based encoder-decoder models with multi-band inputs from Sentinel-2 and Formosat-5 imagery. Our architecture supports multiple decoder variants and multi-stage loss strategies to improve performance under limited supervision. During the evaluation, model predictions are compared with higher-resolution EVAP output to assess spatial coherence and segmentation consistency. Case studies on the 2022 Poyang Lake drought and the 2023 Rhodes wildfire demonstrate that our framework improves the smoothness and reliability of segmentation results, offering a scalable approach for disaster mapping when accurate ground truth is unavailable.

---

## Joint Asymmetric Loss for Learning with Noisy Labels

**Link:** [https://arxiv.org/pdf/2507.17692](https://arxiv.org/pdf/2507.17692)

**Similarity:** 0.1846

### Abstract

Learning with noisy labels is a crucial task for training accurate deep neural networks. To mitigate label noise, prior studies have proposed various robust loss functions, particularly symmetric losses. Nevertheless, symmetric losses usually suffer from the underfitting issue due to the overly strict constraint. To address this problem, the Active Passive Loss (APL) jointly optimizes an active and a passive loss to mutually enhance the overall fitting ability. Within APL, symmetric losses have been successfully extended, yielding advanced robust loss functions. Despite these advancements, emerging theoretical analyses indicate that asymmetric losses, a new class of robust loss functions, possess superior properties compared to symmetric losses. However, existing asymmetric losses are not compatible with advanced optimization frameworks such as APL, limiting their potential and applicability. Motivated by this theoretical gap and the prospect of asymmetric losses, we extend the asymmetric loss to the more complex passive loss scenario and propose the Asymetric Mean Square Error (AMSE), a novel asymmetric loss. We rigorously establish the necessary and sufficient condition under which AMSE satisfies the asymmetric condition. By substituting the traditional symmetric passive loss in APL with our proposed AMSE, we introduce a novel robust loss framework termed Joint Asymmetric Loss (JAL). Extensive experiments demonstrate the effectiveness of our method in mitigating label noise. Code available at: this https URL

---

## Fully Automated SAM for Single-source Domain Generalization in Medical Image Segmentation

**Link:** [https://arxiv.org/pdf/2507.17281](https://arxiv.org/pdf/2507.17281)

**Similarity:** 0.1844

### Abstract

Although SAM-based single-source domain generalization models for medical image segmentation can mitigate the impact of domain shift on the model in cross-domain scenarios, these models still face two major challenges. First, the segmentation of SAM is highly dependent on domain-specific expert-annotated prompts, which prevents SAM from achieving fully automated medical image segmentation and therefore limits its application in clinical settings. Second, providing poor prompts (such as bounding boxes that are too small or too large) to the SAM prompt encoder can mislead SAM into generating incorrect mask results. Therefore, we propose the FA-SAM, a single-source domain generalization framework for medical image segmentation that achieves fully automated SAM. FA-SAM introduces two key innovations: an Auto-prompted Generation Model (AGM) branch equipped with a Shallow Feature Uncertainty Modeling (SUFM) module, and an Image-Prompt Embedding Fusion (IPEF) module integrated into the SAM mask decoder. Specifically, AGM models the uncertainty distribution of shallow features through the SUFM module to generate bounding box prompts for the target domain, enabling fully automated segmentation with SAM. The IPEF module integrates multiscale information from SAM image embeddings and prompt embeddings to capture global and local details of the target object, enabling SAM to mitigate the impact of poor prompts. Extensive experiments on publicly available prostate and fundus vessel datasets validate the effectiveness of FA-SAM and highlight its potential to address the above challenges.

---

## PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving

**Link:** [https://arxiv.org/pdf/2507.17596](https://arxiv.org/pdf/2507.17596)

**Similarity:** 0.1825

### Abstract

While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at this https URL.

---

## Attention (as Discrete-Time Markov) Chains

**Link:** [https://arxiv.org/pdf/2507.17657](https://arxiv.org/pdf/2507.17657)

**Similarity:** 0.1815

### Abstract

We introduce a new interpretation of the attention matrix as a discrete-time Markov chain. Our interpretation sheds light on common operations involving attention scores such as selection, summation, and averaging in a unified framework. It further extends them by considering indirect attention, propagated through the Markov chain, as opposed to previous studies that only model immediate effects. Our main observation is that tokens corresponding to semantically similar regions form a set of metastable states, where the attention clusters, while noisy attention scores tend to disperse. Metastable states and their prevalence can be easily computed through simple matrix multiplication and eigenanalysis, respectively. Using these lightweight tools, we demonstrate state-of-the-art zero-shot segmentation. Lastly, we define TokenRank -- the steady state vector of the Markov chain, which measures global token importance. We demonstrate that using it brings improvements in unconditional image generation. We believe our framework offers a fresh view of how tokens are being attended in modern visual transformers.

---

## CausalStep: A Benchmark for Explicit Stepwise Causal Reasoning in Videos

**Link:** [https://arxiv.org/pdf/2507.16878](https://arxiv.org/pdf/2507.16878)

**Similarity:** 0.1758

### Abstract

Recent advances in large language models (LLMs) have improved reasoning in text and image domains, yet achieving robust video reasoning remains a significant challenge. Existing video benchmarks mainly assess shallow understanding and reasoning and allow models to exploit global context, failing to rigorously evaluate true causal and stepwise reasoning. We present CausalStep, a benchmark designed for explicit stepwise causal reasoning in videos. CausalStep segments videos into causally linked units and enforces a strict stepwise question-answer (QA) protocol, requiring sequential answers and preventing shortcut solutions. Each question includes carefully constructed distractors based on error type taxonomy to ensure diagnostic value. The benchmark features 100 videos across six categories and 1,852 multiple-choice QA pairs. We introduce seven diagnostic metrics for comprehensive evaluation, enabling precise diagnosis of causal reasoning capabilities. Experiments with leading proprietary and open-source models, as well as human baselines, reveal a significant gap between current models and human-level stepwise reasoning. CausalStep provides a rigorous benchmark to drive progress in robust and interpretable video reasoning.

---

## InvRGB+L: Inverse Rendering of Complex Scenes with Unified Color and LiDAR Reflectance Modeling

**Link:** [https://arxiv.org/pdf/2507.17613](https://arxiv.org/pdf/2507.17613)

**Similarity:** 0.1721

### Abstract

We present InvRGB+L, a novel inverse rendering model that reconstructs large, relightable, and dynamic scenes from a single RGB+LiDAR sequence. Conventional inverse graphics methods rely primarily on RGB observations and use LiDAR mainly for geometric information, often resulting in suboptimal material estimates due to visible light interference. We find that LiDAR's intensity values-captured with active illumination in a different spectral range-offer complementary cues for robust material estimation under variable lighting. Inspired by this, InvRGB+L leverages LiDAR intensity cues to overcome challenges inherent in RGB-centric inverse graphics through two key innovations: (1) a novel physics-based LiDAR shading model and (2) RGB-LiDAR material consistency losses. The model produces novel-view RGB and LiDAR renderings of urban and indoor scenes and supports relighting, night simulations, and dynamic object insertions, achieving results that surpass current state-of-the-art methods in both scene-level urban inverse rendering and LiDAR simulation.

---

## STQE: Spatial-Temporal Quality Enhancement for G-PCC Compressed Dynamic Point Clouds

**Link:** [https://arxiv.org/pdf/2507.17522](https://arxiv.org/pdf/2507.17522)

**Similarity:** 0.1714

### Abstract

Very few studies have addressed quality enhancement for compressed dynamic point clouds. In particular, the effective exploitation of spatial-temporal correlations between point cloud frames remains largely unexplored. Addressing this gap, we propose a spatial-temporal attribute quality enhancement (STQE) network that exploits both spatial and temporal correlations to improve the visual quality of G-PCC compressed dynamic point clouds. Our contributions include a recoloring-based motion compensation module that remaps reference attribute information to the current frame geometry to achieve precise inter-frame geometric alignment, a channel-aware temporal attention module that dynamically highlights relevant regions across bidirectional reference frames, a Gaussian-guided neighborhood feature aggregation module that efficiently captures spatial dependencies between geometry and color attributes, and a joint loss function based on the Pearson correlation coefficient, designed to alleviate over-smoothing effects typical of point-wise mean squared error optimization. When applied to the latest G-PCC test model, STQE achieved improvements of 0.855 dB, 0.682 dB, and 0.828 dB in delta PSNR, with Bjøntegaard Delta rate (BD-rate) reductions of -25.2%, -31.6%, and -32.5% for the Luma, Cb, and Cr components, respectively.

---

## Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection

**Link:** [https://arxiv.org/pdf/2507.16861](https://arxiv.org/pdf/2507.16861)

**Similarity:** 0.1695

### Abstract

Integrating LiDAR and camera inputs into a unified Bird's-Eye-View (BEV) representation is crucial for enhancing 3D perception capabilities of autonomous vehicles. However, current methods are often affected by misalignment between camera and LiDAR features. This misalignment leads to inaccurate depth supervision in camera branch and erroneous fusion during cross-modal feature aggregation. The root cause of this misalignment lies in projection errors, stemming from minor extrinsic calibration inaccuracies and rolling shutter effect of LiDAR during vehicle motion. In this work, our key insight is that these projection errors are predominantly concentrated at object-background boundaries, which are readily identified by 2D detectors. Based on this, our main motivation is to utilize 2D object priors to pre-align cross-modal features before fusion. To address local misalignment, we propose Prior Guided Depth Calibration (PGDC), which leverages 2D priors to correct local misalignment and preserve correct cross-modal feature pairs. To resolve global misalignment, we introduce Discontinuity Aware Geometric Fusion (DAGF) to process calibrated results from PGDC, suppressing noise and explicitly enhancing sharp transitions at object-background boundaries. To effectively utilize these transition-aware depth representations, we incorporate Structural Guidance Depth Modulator (SGDM), using a gated attention mechanism to efficiently fuse aligned depth and image features. Our proposed method achieves state-of-the-art performance on nuScenes validation dataset, with its mAP and NDS reaching 71.5% and 73.6% respectively.

---

## UNICE: Training A Universal Image Contrast Enhancer

**Link:** [https://arxiv.org/pdf/2507.17157](https://arxiv.org/pdf/2507.17157)

**Similarity:** 0.1671

### Abstract

Existing image contrast enhancement methods are typically designed for specific tasks such as under-/over-exposure correction, low-light and backlit image enhancement, etc. The learned models, however, exhibit poor generalization performance across different tasks, even across different datasets of a specific task. It is important to explore whether we can learn a universal and generalized model for various contrast enhancement tasks. In this work, we observe that the common key factor of these tasks lies in the need of exposure and contrast adjustment, which can be well-addressed if high-dynamic range (HDR) inputs are available. We hence collect 46,928 HDR raw images from public sources, and render 328,496 sRGB images to build multi-exposure sequences (MES) and the corresponding pseudo sRGB ground-truths via multi-exposure fusion. Consequently, we train a network to generate an MES from a single sRGB image, followed by training another network to fuse the generated MES into an enhanced image. Our proposed method, namely UNiversal Image Contrast Enhancer (UNICE), is free of costly human labeling. However, it demonstrates significantly stronger generalization performance than existing image contrast enhancement methods across and within different tasks, even outperforming manually created ground-truths in multiple no-reference image quality metrics. The dataset, code and model are available at this https URL.

---

## SADA: Stability-guided Adaptive Diffusion Acceleration

**Link:** [https://arxiv.org/pdf/2507.17135](https://arxiv.org/pdf/2507.17135)

**Similarity:** 0.1671

### Abstract

Diffusion models have achieved remarkable success in generative tasks but suffer from high computational costs due to their iterative sampling process and quadratic attention costs. Existing training-free acceleration strategies that reduce per-step computation cost, while effectively reducing sampling time, demonstrate low faithfulness compared to the original baseline. We hypothesize that this fidelity gap arises because (a) different prompts correspond to varying denoising trajectory, and (b) such methods do not consider the underlying ODE formulation and its numerical solution. In this paper, we propose Stability-guided Adaptive Diffusion Acceleration (SADA), a novel paradigm that unifies step-wise and token-wise sparsity decisions via a single stability criterion to accelerate sampling of ODE-based generative models (Diffusion and Flow-matching). For (a), SADA adaptively allocates sparsity based on the sampling trajectory. For (b), SADA introduces principled approximation schemes that leverage the precise gradient information from the numerical ODE solver. Comprehensive evaluations on SD-2, SDXL, and Flux using both EDM and DPM++ solvers reveal consistent $\ge 1.8\times$ speedups with minimal fidelity degradation (LPIPS $\leq 0.10$ and FID $\leq 4.5$) compared to unmodified baselines, significantly outperforming prior methods. Moreover, SADA adapts seamlessly to other pipelines and modalities: It accelerates ControlNet without any modifications and speeds up MusicLDM by $1.8\times$ with $\sim 0.01$ spectrogram LPIPS.

---

## Perspective-Invariant 3D Object Detection

**Link:** [https://arxiv.org/pdf/2507.17665](https://arxiv.org/pdf/2507.17665)

**Similarity:** 0.1670

### Abstract

With the rise of robotics, LiDAR-based 3D object detection has garnered significant attention in both academia and industry. However, existing datasets and methods predominantly focus on vehicle-mounted platforms, leaving other autonomous platforms underexplored. To bridge this gap, we introduce Pi3DET, the first benchmark featuring LiDAR data and 3D bounding box annotations collected from multiple platforms: vehicle, quadruped, and drone, thereby facilitating research in 3D object detection for non-vehicle platforms as well as cross-platform 3D detection. Based on Pi3DET, we propose a novel cross-platform adaptation framework that transfers knowledge from the well-studied vehicle platform to other platforms. This framework achieves perspective-invariant 3D detection through robust alignment at both geometric and feature levels. Additionally, we establish a benchmark to evaluate the resilience and robustness of current 3D detectors in cross-platform scenarios, providing valuable insights for developing adaptive 3D perception systems. Extensive experiments validate the effectiveness of our approach on challenging cross-platform tasks, demonstrating substantial gains over existing adaptation methods. We hope this work paves the way for generalizable and unified 3D perception systems across diverse and complex environments. Our Pi3DET dataset, cross-platform benchmark suite, and annotation toolkit have been made publicly available.

---

## Vision Transformer attention alignment with human visual perception in aesthetic object evaluation

**Link:** [https://arxiv.org/pdf/2507.17616](https://arxiv.org/pdf/2507.17616)

**Similarity:** 0.1652

### Abstract

Visual attention mechanisms play a crucial role in human perception and aesthetic evaluation. Recent advances in Vision Transformers (ViTs) have demonstrated remarkable capabilities in computer vision tasks, yet their alignment with human visual attention patterns remains underexplored, particularly in aesthetic contexts. This study investigates the correlation between human visual attention and ViT attention mechanisms when evaluating handcrafted objects. We conducted an eye-tracking experiment with 30 participants (9 female, 21 male, mean age 24.6 years) who viewed 20 artisanal objects comprising basketry bags and ginger jars. Using a Pupil Labs eye-tracker, we recorded gaze patterns and generated heat maps representing human visual attention. Simultaneously, we analyzed the same objects using a pre-trained ViT model with DINO (Self-DIstillation with NO Labels), extracting attention maps from each of the 12 attention heads. We compared human and ViT attention distributions using Kullback-Leibler divergence across varying Gaussian parameters (sigma=0.1 to 3.0). Statistical analysis revealed optimal correlation at sigma=2.4 +-0.03, with attention head #12 showing the strongest alignment with human visual patterns. Significant differences were found between attention heads, with heads #7 and #9 demonstrating the greatest divergence from human attention (p< 0.05, Tukey HSD test). Results indicate that while ViTs exhibit more global attention patterns compared to human focal attention, certain attention heads can approximate human visual behavior, particularly for specific object features like buckles in basketry items. These findings suggest potential applications of ViT attention mechanisms in product design and aesthetic evaluation, while highlighting fundamental differences in attention strategies between human perception and current AI models.

---

## Vec2Face+ for Face Dataset Generation

**Link:** [https://arxiv.org/pdf/2507.17192](https://arxiv.org/pdf/2507.17192)

**Similarity:** 0.1644

### Abstract

When synthesizing identities as face recognition training data, it is generally believed that large inter-class separability and intra-class attribute variation are essential for synthesizing a quality dataset. % This belief is generally correct, and this is what we aim for. However, when increasing intra-class variation, existing methods overlook the necessity of maintaining intra-class identity consistency. % To address this and generate high-quality face training data, we propose Vec2Face+, a generative model that creates images directly from image features and allows for continuous and easy control of face identities and attributes. Using Vec2Face+, we obtain datasets with proper inter-class separability and intra-class variation and identity consistency using three strategies: 1) we sample vectors sufficiently different from others to generate well-separated identities; 2) we propose an AttrOP algorithm for increasing general attribute variations; 3) we propose LoRA-based pose control for generating images with profile head poses, which is more efficient and identity-preserving than AttrOP. % Our system generates VFace10K, a synthetic face dataset with 10K identities, which allows an FR model to achieve state-of-the-art accuracy on seven real-world test sets. Scaling the size to 4M and 12M images, the corresponding VFace100K and VFace300K datasets yield higher accuracy than the real-world training dataset, CASIA-WebFace, on five real-world test sets. This is the first time a synthetic dataset beats the CASIA-WebFace in average accuracy. In addition, we find that only 1 out of 11 synthetic datasets outperforms random guessing (\emph{i.e., 50\%}) in twin verification and that models trained with synthetic identities are more biased than those trained with real identities. Both are important aspects for future investigation.

---

## See the Forest and the Trees: A Synergistic Reasoning Framework for Knowledge-Based Visual Question Answering

**Link:** [https://arxiv.org/pdf/2507.17659](https://arxiv.org/pdf/2507.17659)

**Similarity:** 0.1633

### Abstract

Multimodal Large Language Models (MLLMs) have pushed the frontiers of Knowledge-Based Visual Question Answering (KBVQA), yet their reasoning is fundamentally bottlenecked by a reliance on uni-dimensional evidence. This "seeing only the trees, but not the forest" approach prevents robust, multi-faceted understanding. Inspired by the principle of seeing both the forest and trees, we propose Synergos-VQA, a novel synergistic reasoning framework. At its core, Synergos-VQA concurrently generates and fuses three complementary evidence streams at inference time: (1) Holistic Evidence to perceive the entire scene (the "forest"), (2) Structural Evidence from a prototype-driven module to identify key objects (the "trees"), and (3) Causal Evidence from a counterfactual probe to ensure the reasoning is robustly grounded. By synergistically fusing this multi-faceted evidence, our framework achieves a more comprehensive and reliable reasoning process. Extensive experiments show that Synergos-VQA decisively establishes a new state-of-the-art on three challenging benchmarks, including OK-VQA and A-OKVQA. Furthermore, our approach demonstrates strong plug-and-play capabilities, significantly boosting various open-source MLLMs and proving that superior methodological design can outperform sheer model scale.

---

## Temporal Point-Supervised Signal Reconstruction: A Human-Annotation-Free Framework for Weak Moving Target Detection

**Link:** [https://arxiv.org/pdf/2507.17334](https://arxiv.org/pdf/2507.17334)

**Similarity:** 0.1626

### Abstract

In low-altitude surveillance and early warning systems, detecting weak moving targets remains a significant challenge due to low signal energy, small spatial extent, and complex background clutter. Existing methods struggle with extracting robust features and suffer from the lack of reliable annotations. To address these limitations, we propose a novel Temporal Point-Supervised (TPS) framework that enables high-performance detection of weak targets without any manual this http URL of conventional frame-based detection, our framework reformulates the task as a pixel-wise temporal signal modeling problem, where weak targets manifest as short-duration pulse-like responses. A Temporal Signal Reconstruction Network (TSRNet) is developed under the TPS paradigm to reconstruct these transient this http URL adopts an encoder-decoder architecture and integrates a Dynamic Multi-Scale Attention (DMSAttention) module to enhance its sensitivity to diverse temporal patterns. Additionally, a graph-based trajectory mining strategy is employed to suppress false alarms and ensure temporal this http URL experiments on a purpose-built low-SNR dataset demonstrate that our framework outperforms state-of-the-art methods while requiring no human annotations. It achieves strong detection performance and operates at over 1000 FPS, underscoring its potential for real-time deployment in practical scenarios.

---

## Dual-branch Prompting for Multimodal Machine Translation

**Link:** [https://arxiv.org/pdf/2507.17588](https://arxiv.org/pdf/2507.17588)

**Similarity:** 0.1620

### Abstract

Multimodal Machine Translation (MMT) typically enhances text-only translation by incorporating aligned visual features. Despite the remarkable progress, state-of-the-art MMT approaches often rely on paired image-text inputs at inference and are sensitive to irrelevant visual noise, which limits their robustness and practical applicability. To address these issues, we propose D2P-MMT, a diffusion-based dual-branch prompting framework for robust vision-guided translation. Specifically, D2P-MMT requires only the source text and a reconstructed image generated by a pre-trained diffusion model, which naturally filters out distracting visual details while preserving semantic cues. During training, the model jointly learns from both authentic and reconstructed images using a dual-branch prompting strategy, encouraging rich cross-modal interactions. To bridge the modality gap and mitigate training-inference discrepancies, we introduce a distributional alignment loss that enforces consistency between the output distributions of the two branches. Extensive experiments on the Multi30K dataset demonstrate that D2P-MMT achieves superior translation performance compared to existing state-of-the-art approaches.

---

## SDGOCC: Semantic and Depth-Guided Bird's-Eye View Transformation for 3D Multimodal Occupancy Prediction

**Link:** [https://arxiv.org/pdf/2507.17083](https://arxiv.org/pdf/2507.17083)

**Similarity:** 0.1618

### Abstract

Multimodal 3D occupancy prediction has garnered significant attention for its potential in autonomous driving. However, most existing approaches are single-modality: camera-based methods lack depth information, while LiDAR-based methods struggle with occlusions. Current lightweight methods primarily rely on the Lift-Splat-Shoot (LSS) pipeline, which suffers from inaccurate depth estimation and fails to fully exploit the geometric and semantic information of 3D LiDAR points. Therefore, we propose a novel multimodal occupancy prediction network called SDG-OCC, which incorporates a joint semantic and depth-guided view transformation coupled with a fusion-to-occupancy-driven active distillation. The enhanced view transformation constructs accurate depth distributions by integrating pixel semantics and co-point depth through diffusion and bilinear discretization. The fusion-to-occupancy-driven active distillation extracts rich semantic information from multimodal data and selectively transfers knowledge to image features based on LiDAR-identified regions. Finally, for optimal performance, we introduce SDG-Fusion, which uses fusion alone, and SDG-KL, which integrates both fusion and distillation for faster inference. Our method achieves state-of-the-art (SOTA) performance with real-time processing on the Occ3D-nuScenes dataset and shows comparable performance on the more challenging SurroundOcc-nuScenes dataset, demonstrating its effectiveness and robustness. The code will be released at this https URL.

---

## An h-space Based Adversarial Attack for Protection Against Few-shot Personalization

**Link:** [https://arxiv.org/pdf/2507.17554](https://arxiv.org/pdf/2507.17554)

**Similarity:** 0.1585

### Abstract

The versatility of diffusion models in generating customized images from few samples raises significant privacy concerns, particularly regarding unauthorized modifications of private content. This concerning issue has renewed the efforts in developing protection mechanisms based on adversarial attacks, which generate effective perturbations to poison diffusion models. Our work is motivated by the observation that these models exhibit a high degree of abstraction within their semantic latent space (`h-space'), which encodes critical high-level features for generating coherent and meaningful content. In this paper, we propose a novel anti-customization approach, called HAAD (h-space based Adversarial Attack for Diffusion models), that leverages adversarial attacks to craft perturbations based on the h-space that can efficiently degrade the image generation process. Building upon HAAD, we further introduce a more efficient variant, HAAD-KV, that constructs perturbations solely based on the KV parameters of the h-space. This strategy offers a stronger protection, that is computationally less expensive. Despite their simplicity, our methods outperform state-of-the-art adversarial attacks, highlighting their effectiveness.

---

## Explainable AI for Collaborative Assessment of 2D/3D Registration Quality

**Link:** [https://arxiv.org/pdf/2507.17597](https://arxiv.org/pdf/2507.17597)

**Similarity:** 0.1576

### Abstract

As surgery embraces digital transformation--integrating sophisticated imaging, advanced algorithms, and robotics to support and automate complex sub-tasks--human judgment of system correctness remains a vital safeguard for patient safety. This shift introduces new "operator-type" roles tasked with verifying complex algorithmic outputs, particularly at critical junctures of the procedure, such as the intermediary check before drilling or implant placement. A prime example is 2D/3D registration, a key enabler of image-based surgical navigation that aligns intraoperative 2D images with preoperative 3D data. Although registration algorithms have advanced significantly, they occasionally yield inaccurate results. Because even small misalignments can lead to revision surgery or irreversible surgical errors, there is a critical need for robust quality assurance. Current visualization-based strategies alone have been found insufficient to enable humans to reliably detect 2D/3D registration misalignments. In response, we propose the first artificial intelligence (AI) framework trained specifically for 2D/3D registration quality verification, augmented by explainability features that clarify the model's decision-making. Our explainable AI (XAI) approach aims to enhance informed decision-making for human operators by providing a second opinion together with a rationale behind it. Through algorithm-centric and human-centered evaluations, we systematically compare four conditions: AI-only, human-only, human-AI, and human-XAI. Our findings reveal that while explainability features modestly improve user trust and willingness to override AI errors, they do not exceed the standalone AI in aggregate performance. Nevertheless, future work extending both the algorithmic design and the human-XAI collaboration elements holds promise for more robust quality assurance of 2D/3D registration.

---

## DOOMGAN:High-Fidelity Dynamic Identity Obfuscation Ocular Generative Morphing

**Link:** [https://arxiv.org/pdf/2507.17158](https://arxiv.org/pdf/2507.17158)

**Similarity:** 0.1527

### Abstract

Ocular biometrics in the visible spectrum have emerged as a prominent modality due to their high accuracy, resistance to spoofing, and non-invasive nature. However, morphing attacks, synthetic biometric traits created by blending features from multiple individuals, threaten biometric system integrity. While extensively studied for near-infrared iris and face biometrics, morphing in visible-spectrum ocular data remains underexplored. Simulating such attacks demands advanced generation models that handle uncontrolled conditions while preserving detailed ocular features like iris boundaries and periocular textures. To address this gap, we introduce DOOMGAN, that encompasses landmark-driven encoding of visible ocular anatomy, attention-guided generation for realistic morph synthesis, and dynamic weighting of multi-faceted losses for optimized convergence. DOOMGAN achieves over 20% higher attack success rates than baseline methods under stringent thresholds, along with 20% better elliptical iris structure generation and 30% improved gaze consistency. We also release the first comprehensive ocular morphing dataset to support further research in this domain.

---

## Few-Shot Learning in Video and 3D Object Detection: A Survey

**Link:** [https://arxiv.org/pdf/2507.17079](https://arxiv.org/pdf/2507.17079)

**Similarity:** 0.1516

### Abstract

Few-shot learning (FSL) enables object detection models to recognize novel classes given only a few annotated examples, thereby reducing expensive manual data labeling. This survey examines recent FSL advances for video and 3D object detection. For video, FSL is especially valuable since annotating objects across frames is more laborious than for static images. By propagating information across frames, techniques like tube proposals and temporal matching networks can detect new classes from a couple examples, efficiently leveraging spatiotemporal structure. FSL for 3D detection from LiDAR or depth data faces challenges like sparsity and lack of texture. Solutions integrate FSL with specialized point cloud networks and losses tailored for class imbalance. Few-shot 3D detection enables practical autonomous driving deployment by minimizing costly 3D annotation needs. Core issues in both domains include balancing generalization and overfitting, integrating prototype matching, and handling data modality properties. In summary, FSL shows promise for reducing annotation requirements and enabling real-world video, 3D, and other applications by efficiently leveraging information across feature, temporal, and data modalities. By comprehensively surveying recent advancements, this paper illuminates FSL's potential to minimize supervision needs and enable deployment across video, 3D, and other real-world applications.

---

## TransLPRNet: Lite Vision-Language Network for Single/Dual-line Chinese License Plate Recognition

**Link:** [https://arxiv.org/pdf/2507.17335](https://arxiv.org/pdf/2507.17335)

**Similarity:** 0.1493

### Abstract

License plate recognition in open environments is widely applicable across various domains; however, the diversity of license plate types and imaging conditions presents significant challenges. To address the limitations encountered by CNN and CRNN-based approaches in license plate recognition, this paper proposes a unified solution that integrates a lightweight visual encoder with a text decoder, within a pre-training framework tailored for single and double-line Chinese license plates. To mitigate the scarcity of double-line license plate datasets, we constructed a single/double-line license plate dataset by synthesizing images, applying texture mapping onto real scenes, and blending them with authentic license plate images. Furthermore, to enhance the system's recognition accuracy, we introduce a perspective correction network (PTN) that employs license plate corner coordinate regression as an implicit variable, supervised by license plate view classification information. This network offers improved stability, interpretability, and low annotation costs. The proposed algorithm achieves an average recognition accuracy of 99.34% on the corrected CCPD test set under coarse localization disturbance. When evaluated under fine localization disturbance, the accuracy further improves to 99.58%. On the double-line license plate test set, it achieves an average recognition accuracy of 98.70%, with processing speeds reaching up to 167 frames per second, indicating strong practical applicability.

---

## URPO: A Unified Reward & Policy Optimization Framework for Large Language Models

**Link:** [https://arxiv.org/pdf/2507.17515](https://arxiv.org/pdf/2507.17515)

**Similarity:** 0.1409

### Abstract

Large-scale alignment pipelines typically pair a policy model with a separately trained reward model whose parameters remain frozen during reinforcement learning (RL). This separation creates a complex, resource-intensive pipeline and suffers from a performance ceiling due to a static reward signal. We propose a novel framework, Unified Reward & Policy Optimization (URPO), that unifies instruction-following ("player") and reward modeling ("referee") within a single model and a single training phase. Our method recasts all alignment data-including preference pairs, verifiable reasoning, and open-ended instructions-into a unified generative format optimized by a single Group-Relative Policy Optimization (GRPO) loop. This enables the model to learn from ground-truth preferences and verifiable logic while simultaneously generating its own rewards for open-ended tasks. Experiments on the Qwen2.5-7B model demonstrate URPO's superiority. Our unified model significantly outperforms a strong baseline using a separate generative reward model, boosting the instruction-following score on AlpacaEval from 42.24 to 44.84 and the composite reasoning average from 32.66 to 35.66. Furthermore, URPO cultivates a superior internal evaluator as a byproduct of training, achieving a RewardBench score of 85.15 and surpassing the dedicated reward model it replaces (83.55). By eliminating the need for a separate reward model and fostering a co-evolutionary dynamic between generation and evaluation, URPO presents a simpler, more efficient, and more effective path towards robustly aligned language models.

---

## HIPPO-Video: Simulating Watch Histories with Large Language Models for Personalized Video Highlighting

**Link:** [https://arxiv.org/pdf/2507.16873](https://arxiv.org/pdf/2507.16873)

**Similarity:** 0.1368

### Abstract

The exponential growth of video content has made personalized video highlighting an essential task, as user preferences are highly variable and complex. Existing video datasets, however, often lack personalization, relying on isolated videos or simple text queries that fail to capture the intricacies of user behavior. In this work, we introduce HIPPO-Video, a novel dataset for personalized video highlighting, created using an LLM-based user simulator to generate realistic watch histories reflecting diverse user preferences. The dataset includes 2,040 (watch history, saliency score) pairs, covering 20,400 videos across 170 semantic categories. To validate our dataset, we propose HiPHer, a method that leverages these personalized watch histories to predict preference-conditioned segment-wise saliency scores. Through extensive experiments, we demonstrate that our method outperforms existing generic and query-based approaches, showcasing its potential for highly user-centric video highlighting in real-world scenarios.

---

## Dataset Distillation as Data Compression: A Rate-Utility Perspective

**Link:** [https://arxiv.org/pdf/2507.17221](https://arxiv.org/pdf/2507.17221)

**Similarity:** 0.1362

### Abstract

Driven by the ``scale-is-everything'' paradigm, modern machine learning increasingly demands ever-larger datasets and models, yielding prohibitive computational and storage requirements. Dataset distillation mitigates this by compressing an original dataset into a small set of synthetic samples, while preserving its full utility. Yet, existing methods either maximize performance under fixed storage budgets or pursue suitable synthetic data representations for redundancy removal, without jointly optimizing both objectives. In this work, we propose a joint rate-utility optimization method for dataset distillation. We parameterize synthetic samples as optimizable latent codes decoded by extremely lightweight networks. We estimate the Shannon entropy of quantized latents as the rate measure and plug any existing distillation loss as the utility measure, trading them off via a Lagrange multiplier. To enable fair, cross-method comparisons, we introduce bits per class (bpc), a precise storage metric that accounts for sample, label, and decoder parameter costs. On CIFAR-10, CIFAR-100, and ImageNet-128, our method achieves up to $170\times$ greater compression than standard distillation at comparable accuracy. Across diverse bpc budgets, distillation losses, and backbone architectures, our approach consistently establishes better rate-utility trade-offs.

---

## Integrating Belief Domains into Probabilistic Logic Programs

**Link:** [https://arxiv.org/pdf/2507.17291](https://arxiv.org/pdf/2507.17291)

**Similarity:** 0.1357

### Abstract

Probabilistic Logic Programming (PLP) under the Distribution Semantics is a leading approach to practical reasoning under uncertainty. An advantage of the Distribution Semantics is its suitability for implementation as a Prolog or Python library, available through two well-maintained implementations, namely ProbLog and cplint/PITA. However, current formulations of the Distribution Semantics use point-probabilities, making it difficult to express epistemic uncertainty, such as arises from, for example, hierarchical classifications from computer vision models. Belief functions generalize probability measures as non-additive capacities, and address epistemic uncertainty via interval probabilities. This paper introduces interval-based Capacity Logic Programs based on an extension of the Distribution Semantics to include belief functions, and describes properties of the new framework that make it amenable to practical applications.

---

## VBCD: A Voxel-Based Framework for Personalized Dental Crown Design

**Link:** [https://arxiv.org/pdf/2507.17205](https://arxiv.org/pdf/2507.17205)

**Similarity:** 0.1328

### Abstract

The design of restorative dental crowns from intraoral scans is labor-intensive for dental technicians. To address this challenge, we propose a novel voxel-based framework for automated dental crown design (VBCD). The VBCD framework generates an initial coarse dental crown from voxelized intraoral scans, followed by a fine-grained refiner incorporating distance-aware supervision to improve accuracy and quality. During the training stage, we employ the Curvature and Margin line Penalty Loss (CMPL) to enhance the alignment of the generated crown with the margin line. Additionally, a positional prompt based on the FDI tooth numbering system is introduced to further improve the accuracy of the generated dental crowns. Evaluation on a large-scale dataset of intraoral scans demonstrated that our approach outperforms existing methods, providing a robust solution for personalized dental crown design.

---

## PARTE: Part-Guided Texturing for 3D Human Reconstruction from a Single Image

**Link:** [https://arxiv.org/pdf/2507.17332](https://arxiv.org/pdf/2507.17332)

**Similarity:** 0.1321

### Abstract

The misaligned human texture across different human parts is one of the main limitations of existing 3D human reconstruction methods. Each human part, such as a jacket or pants, should maintain a distinct texture without blending into others. The structural coherence of human parts serves as a crucial cue to infer human textures in the invisible regions of a single image. However, most existing 3D human reconstruction methods do not explicitly exploit such part segmentation priors, leading to misaligned textures in their reconstructions. In this regard, we present PARTE, which utilizes 3D human part information as a key guide to reconstruct 3D human textures. Our framework comprises two core components. First, to infer 3D human part information from a single image, we propose a 3D part segmentation module (PartSegmenter) that initially reconstructs a textureless human surface and predicts human part labels based on the textureless surface. Second, to incorporate part information into texture reconstruction, we introduce a part-guided texturing module (PartTexturer), which acquires prior knowledge from a pre-trained image generation network on texture alignment of human parts. Extensive experiments demonstrate that our framework achieves state-of-the-art quality in 3D human reconstruction. The project page is available at this https URL.

---

## VisionTrap: Unanswerable Questions On Visual Data

**Link:** [https://arxiv.org/pdf/2507.17262](https://arxiv.org/pdf/2507.17262)

**Similarity:** 0.1283

### Abstract

Visual Question Answering (VQA) has been a widely studied topic, with extensive research focusing on how VLMs respond to answerable questions based on real-world images. However, there has been limited exploration of how these models handle unanswerable questions, particularly in cases where they should abstain from providing a response. This research investigates VQA performance on unrealistically generated images or asking unanswerable questions, assessing whether models recognize the limitations of their knowledge or attempt to generate incorrect answers. We introduced a dataset, VisionTrap, comprising three categories of unanswerable questions across diverse image types: (1) hybrid entities that fuse objects and animals, (2) objects depicted in unconventional or impossible scenarios, and (3) fictional or non-existent figures. The questions posed are logically structured yet inherently unanswerable, testing whether models can correctly recognize their limitations. Our findings highlight the importance of incorporating such questions into VQA benchmarks to evaluate whether models tend to answer, even when they should abstain.

---

## SFUOD: Source-Free Unknown Object Detection

**Link:** [https://arxiv.org/pdf/2507.17373](https://arxiv.org/pdf/2507.17373)

**Similarity:** 0.1266

### Abstract

Source-free object detection adapts a detector pre-trained on a source domain to an unlabeled target domain without requiring access to labeled source data. While this setting is practical as it eliminates the need for the source dataset during domain adaptation, it operates under the restrictive assumption that only pre-defined objects from the source domain exist in the target domain. This closed-set setting prevents the detector from detecting undefined objects. To ease this assumption, we propose Source-Free Unknown Object Detection (SFUOD), a novel scenario which enables the detector to not only recognize known objects but also detect undefined objects as unknown objects. To this end, we propose CollaPAUL (Collaborative tuning and Principal Axis-based Unknown Labeling), a novel framework for SFUOD. Collaborative tuning enhances knowledge adaptation by integrating target-dependent knowledge from the auxiliary encoder with source-dependent knowledge from the pre-trained detector through a cross-domain attention mechanism. Additionally, principal axes-based unknown labeling assigns pseudo-labels to unknown objects by estimating objectness via principal axes projection and confidence scores from model predictions. The proposed CollaPAUL achieves state-of-the-art performances on SFUOD benchmarks, and extensive experiments validate its effectiveness.

---

## PointLAMA: Latent Attention meets Mamba for Efficient Point Cloud Pretraining

**Link:** [https://arxiv.org/pdf/2507.17296](https://arxiv.org/pdf/2507.17296)

**Similarity:** 0.1255

### Abstract

Mamba has recently gained widespread attention as a backbone model for point cloud modeling, leveraging a state-space architecture that enables efficient global sequence modeling with linear complexity. However, its lack of local inductive bias limits its capacity to capture fine-grained geometric structures in 3D data. To address this limitation, we propose \textbf{PointLAMA}, a point cloud pretraining framework that combines task-aware point cloud serialization, a hybrid encoder with integrated Latent Attention and Mamba blocks, and a conditional diffusion mechanism built upon the Mamba backbone. Specifically, the task-aware point cloud serialization employs Hilbert/Trans-Hilbert space-filling curves and axis-wise sorting to structurally align point tokens for classification and segmentation tasks, respectively. Our lightweight Latent Attention block features a Point-wise Multi-head Latent Attention (PMLA) module, which is specifically designed to align with the Mamba architecture by leveraging the shared latent space characteristics of PMLA and Mamba. This enables enhanced local context modeling while preserving overall efficiency. To further enhance representation learning, we incorporate a conditional diffusion mechanism during pretraining, which denoises perturbed feature sequences without relying on explicit point-wise reconstruction. Experimental results demonstrate that PointLAMA achieves competitive performance on multiple benchmark datasets with minimal parameter count and FLOPs, validating its effectiveness for efficient point cloud pretraining.

---

## Hierarchical Fusion and Joint Aggregation: A Multi-Level Feature Representation Method for AIGC Image Quality Assessment

**Link:** [https://arxiv.org/pdf/2507.17182](https://arxiv.org/pdf/2507.17182)

**Similarity:** 0.1170

### Abstract

The quality assessment of AI-generated content (AIGC) faces multi-dimensional challenges, that span from low-level visual perception to high-level semantic understanding. Existing methods generally rely on single-level visual features, limiting their ability to capture complex distortions in AIGC images. To address this limitation, a multi-level visual representation paradigm is proposed with three stages, namely multi-level feature extraction, hierarchical fusion, and joint aggregation. Based on this paradigm, two networks are developed. Specifically, the Multi-Level Global-Local Fusion Network (MGLF-Net) is designed for the perceptual quality assessment, extracting complementary local and global features via dual CNN and Transformer visual backbones. The Multi-Level Prompt-Embedded Fusion Network (MPEF-Net) targets Text-to-Image correspondence by embedding prompt semantics into the visual feature fusion process at each feature level. The fused multi-level features are then aggregated for final evaluation. Experiments on benchmarks demonstrate outstanding performance on both tasks, validating the effectiveness of the proposed multi-level visual assessment paradigm.

---

## PolarAnything: Diffusion-based Polarimetric Image Synthesis

**Link:** [https://arxiv.org/pdf/2507.17268](https://arxiv.org/pdf/2507.17268)

**Similarity:** 0.1154

### Abstract

Polarization images facilitate image enhancement and 3D reconstruction tasks, but the limited accessibility of polarization cameras hinders their broader application. This gap drives the need for synthesizing photorealistic polarization this http URL existing polarization simulator Mitsuba relies on a parametric polarization image formation model and requires extensive 3D assets covering shape and PBR materials, preventing it from generating large-scale photorealistic images. To address this problem, we propose PolarAnything, capable of synthesizing polarization images from a single RGB input with both photorealism and physical accuracy, eliminating the dependency on 3D asset collections. Drawing inspiration from the zero-shot performance of pretrained diffusion models, we introduce a diffusion-based generative framework with an effective representation strategy that preserves the fidelity of polarization properties. Experiments show that our model generates high-quality polarization images and supports downstream tasks like shape from polarization.

---

## Accelerating Parallel Diffusion Model Serving with Residual Compression

**Link:** [https://arxiv.org/pdf/2507.17511](https://arxiv.org/pdf/2507.17511)

**Similarity:** 0.0586

### Abstract

Diffusion models produce realistic images and videos but require substantial computational resources, necessitating multi-accelerator parallelism for real-time deployment. However, parallel inference introduces significant communication overhead from exchanging large activations between devices, limiting efficiency and scalability. We present CompactFusion, a compression framework that significantly reduces communication while preserving generation quality. Our key observation is that diffusion activations exhibit strong temporal redundancy-adjacent steps produce highly similar activations, saturating bandwidth with near-duplicate data carrying little new information. To address this inefficiency, we seek a more compact representation that encodes only the essential information. CompactFusion achieves this via Residual Compression that transmits only compressed residuals (step-wise activation differences). Based on empirical analysis and theoretical justification, we show that it effectively removes redundant data, enabling substantial data reduction while maintaining high fidelity. We also integrate lightweight error feedback to prevent error accumulation. CompactFusion establishes a new paradigm for parallel diffusion inference, delivering lower latency and significantly higher generation quality than prior methods. On 4xL20, it achieves 3.0x speedup while greatly improving fidelity. It also uniquely supports communication-heavy strategies like sequence parallelism on slow networks, achieving 6.7x speedup over prior overlap-based method. CompactFusion applies broadly across diffusion models and parallel settings, and integrates easily without requiring pipeline rework. Portable implementation demonstrated on xDiT is publicly available at this https URL

---

