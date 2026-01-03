# Self-Supervised CLIP-Guided for Few-Shot Industrial Anomaly Detection（SCF-AD）
The code will be released after the manuscript review is completed.

# Abstract
Few-shot industrial anomaly detection aims to identify unseen defects using only a limited number of normal samples. However, most existing approaches still rely heavily on auxiliary industrial datasets for training. In this paper, we propose a novel self-supervised CLIP-guided for few-shot industrial anomaly detection, which eliminates the need for auxiliary industrial data. Specifically, we first introduce a pseudo-anomaly generation strategy that synthesizes both structural and textural anomalies. Then, leveraging the cross-modal semantic understanding capability of CLIP, we contrast the multi-scale visual features with learnable textual prompts to achieve anomaly localization grounded in language semantics. Inspired by the human cognitive process of identifying anomalies through reference comparison, we introduce a support set composed of a few normal samples and perform semantic-level feature alignment with the query set via CLIP visual encoder, thereby enhancing anomaly discrimination. Furthermore, we also introduce Adapter to alleviate the semantic offset problem between text and image modalities in industrial scenarios of CLIP, and enhance the model’s robustness to the spatial structure differences between query set and support set. Extensive experiments conducted on the MVTec AD, the VisA, the BTAD and the MPDD datasets demonstrate that our method achieves competitive results under the few-shot setting. Moreover, its effectiveness and deployability are validated through realworld application in battery spot-welding defect inspection.

# 📊 BSW AD Dataset

The **BSW AD Dataset** is now available for academic research purposes.

## 🔍 How to Apply for Access

Please follow the steps below to request access to the dataset:

1. **Use an academic email address**  
   Send an application email using your institutional email account (e.g., `.edu`, `.stu`, etc.) to:  
   📧 **yikuizhai@163.com**

2. **Sign the usage agreement**  
   - The dataset must **only be used for scientific research**, not for any commercial purposes.  
   - A **scanned copy of the agreement with a handwritten signature** is required.  
   - Signatures in **either Chinese or English** are accepted.

3. **Authorization timeline**  
   - Access will typically be granted within **1–3 business days**.

> ⚠️ **Important Notice**  
> If you use the **BSW AD Dataset** as a **benchmark dataset** in your paper, **please cite the corresponding paper**.

