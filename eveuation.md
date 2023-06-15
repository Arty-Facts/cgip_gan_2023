can you give me an extensive summery of Modified Inception Score (m-IS). In the summary you should state the year a metric was introduced, the basic idea, where it can be used, the formula for the caculation, why its good, and what is the limitation of the metric and add some history of the metric. Can you provide the arxiv.org links in the text if you find any? 

can you present this in  a script window so that i can copy it . use markdown format and use latex for math

# Adversarial Accuracy and Adversarial Divergence

- **Adversarial Accuracy** is a metric for evaluating the robustness of a machine learning model to adversarial attacks. The exact year of introduction is challenging to pinpoint as it has naturally evolved with the emergence of adversarial attacks and the corresponding need for evaluating model robustness.

- **Basic Idea**: Adversarial accuracy quantifies the number of data points for which successful adversarial perturbations cannot be found, given a specific threat model. It measures the model's ability to correctly classify examples that have been slightly manipulated to trick the model into making an error.

- **Advantages**: It offers a quantifiable measure of a model's vulnerability to adversarial attacks. A model with higher adversarial accuracy is typically considered more robust against these kinds of attacks.

- **Adversarial Divergence** is the concept of adversarial divergence was used in a paper titled "Improving Adversarial Robustness by Enforcing Local and Global Compactness" from ECCV 2020.

- **Basic Idea**: The paper proposes the Adversary Divergence Reduction Network, which enforces local/global compactness and the clustering assumption over an intermediate layer of a deep neural network.

- **Advantages**: By enforcing local/global compactness and the clustering assumption, the model can achieve higher unperturbed and adversarial predictive performances.

- **Limitations**: Adversarial accuracy and divergence doesn't capture all aspects of robustness.

**Related Links**: 
* [Improving Adversarial Robustness by Enforcing Local and Global Compactness](https://arxiv.org/abs/2007.05123)
* [Metrics and methods for robustness evaluation of neural networks with generative models](https://arxiv.org/abs/2003.01993)

# Classifier Two-sample Tests (C2ST)

- **Basic Idea:** The Classifier Two-sample Tests metric is a statistical measure used to compare the distributions of two datasets. It assesses whether two datasets, often referred to as the "source" and "target" datasets, are drawn from the same underlying distribution or different distributions. The metric utilizes a classifier to distinguish between the two datasets based on their features and quantifies the statistical discrepancy between the distributions.
- **Usage:** The Classifier Two-sample Tests metric finds application in various domains, including machine learning, computer vision, and domain adaptation. It is particularly useful in evaluating the effectiveness of domain adaptation techniques, where the goal is to transfer knowledge from a source domain to a target domain. The metric helps determine the similarity between the distributions of the source and target domains, aiding in assessing the feasibility of knowledge transfer.
- **Advantages:** The Classifier Two-sample Tests metric offers several advantages for comparing distributions. It provides a principled and statistically grounded approach to assess the similarity or dissimilarity between datasets. The use of a classifier allows for leveraging the discriminative power of the features, enabling more effective discrimination between the datasets. The metric can handle high-dimensional data and is robust to noise and variations within the datasets.
- **Limitations:** The Classifier Two-sample Tests metric has a few limitations. It assumes that the features used for classification adequately capture the differences between the distributions. If the chosen features are not representative or informative, the metric may provide misleading results. Additionally, the metric requires labeled data for training the classifier, which may be a limitation in scenarios where labeled data is scarce or costly to obtain. It is important to consider these limitations and select appropriate features and classifiers for accurate and reliable results.

**Related Links**: 
* [Revisiting Classifier Two-Sample Tests](https://arxiv.org/abs/1610.06545)


# Frechet Inception Distance (FID)

- **Year Introduced:** 2017
- **Basic Idea:** The Frechet Inception Distance (FID) metric is a performance measure used to evaluate the quality of generative models, particularly in the field of image synthesis and generation. It quantifies the similarity between the generated images and real images by comparing their feature representations extracted from a pre-trained deep convolutional neural network (CNN). The Fréchet Inception Distance uses the Inception v3 model, which has been pre-trained on ImageNet, to extract features from an intermediate layer of the network. These features are then used to construct a multivariate Gaussian distribution for both the real and the generated images. The FID is then calculated as the Fréchet distance between these two Gaussian distributions.

- **Theory**

  The Frechet Inception Distance (FID) calculates the distance between the distribution of the Inception features (activations from one of the last layers of the Inception model) of real and generated images.

  The FID is based on the Frechet distance, a measure of similarity between two probability distributions. In the case of FID, it assumes that the Inception features of the images follow a multivariate Gaussian distribution.

  The formula for the FID is as follows:

  $$\text{FID} = ||\mu_1 - \mu_2||^2 + \text{Tr}(\sigma_1 + \sigma_2 - 2(\sigma_1\sigma_2)^{1/2})$$

  where:

  - $\mu_1$ and $\mu_2$ are the means of the Inception features of the real and generated images, respectively,
  - $\sigma_1$ and $\sigma_2$ are the covariance matrices of the Inception features of the real and generated images, respectively,
  - $\text{Tr}$ is the trace of a matrix (the sum of its diagonal elements),
  - $(\sigma_1\sigma_2)^{1/2}$ is the square root of the product of the two covariance matrices.

- **Usage:** The FID metric finds extensive application in evaluating generative models, such as generative adversarial networks (GANs). It is used to compare different models, assess their progress during training, and select the best-performing model.
- **Advantages:** The FID metric offers several advantages for evaluating generative models. It captures both the quality and diversity of the generated images by considering the distribution of the feature representations. The metric is based on a pre-trained CNN, which allows for leveraging high-level semantic information and avoiding the need for manual feature engineering. It provides a single scalar value that reflects the overall quality of the generated images, enabling easy comparison and selection of generative models.
  - It takes into account the diversity of the images generated by the GAN, as well as their quality. A GAN that generates very high-quality images of the same object would have a high FID, as would a GAN that generates a wide variety of poor-quality images
  - It is relatively robust to small changes in the GAN or the dataset. Small improvements in the quality of the generated images will result in small decreases in the FID, and vice versa.

- **Limitations:** 
  - **Reliance on the Inception model:** FID is based on features from the Inception v3 model, which is trained on the ImageNet dataset. This could potentially limit its effectiveness for datasets that are fundamentally different from ImageNet. For example, if the GAN is generating images that are very different from those seen in ImageNet (like medical images or satellite images), the Inception model might not provide a meaningful feature representation, and therefore the FID could be misleading.
  - **Assumption of Gaussian distributions:** FID assumes that the feature representations of real and generated images follow a Gaussian distribution. This might not always be the case, and deviations from this assumption could impact the accuracy of the FID score.
  - **Doesn't capture all aspects of image quality:** While FID takes into account both the quality and diversity of the generated images, it doesn't necessarily capture all aspects of image quality. For example, it might not fully capture perceptual differences such as color or texture, or higher-level attributes such as whether an object in an image looks realistic.
  - **Lack of interpretability:** The FID score is not easy to interpret on its own. It does not have a clear range of values, and lower scores are better, but beyond that, there's no clear meaning attached to specific values. A lower FID indicates better image quality and diversity, but it's hard to say how much "better" an FID of, say, 10 is compared to an FID of 20.
  - **Not sensitive to small localized differences:** FID may not be sensitive to small localized differences or distortions in images, as it operates on summary statistics of the feature distributions. Thus, it may miss subtle but important differences between generated and real images.
  - **Samples size efects the result:** The size of the samples used to compute FID has an impact on the resulting value. Smaller sample sizes may lead to a larger FID and thus less accurate estimates of the feature distributions. It is generally recommended to use a sufficiently large number of samples to obtain reliable and representative feature distributions for accurate FID computation.

**Related Links**: 
* [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
* [Coursera - Fréchet Inception Distance (FID)](https://www.coursera.org/lecture/build-better-generative-adversarial-networks-gans/frechet-inception-distance-fid-LY8WK)


# Generative Adversarial Metric (GAM)

- **Year Introduced:** 2016

- **Basic Idea:** The Generative Adversarial Metric (GAM) is a metric introduced in 2016 to quantitatively compare generative adversarial models (GANs). GANs consist of a generator and a discriminator, and GAM proposes evaluating the models by having them engage in a "battle" against each other. The metric compares the performance of two GANs by examining the discrimination scores of their respective discriminators on both training and test data.

- **Usage:** GAM provides an alternative way to evaluate GANs, allowing direct comparison between different models. It aims to address the challenge of quantitatively evaluating GANs, as existing evaluation methods such as nearest-neighbor data or human inspection have limitations. GAM can be used to assess the generalization ability and sample generation quality of GANs.

- **Advantages:** GAM offers a direct comparison between GAN models, enabling a quantitative assessment of their performance. It considers both the discrimination scores on test data (reflecting generalization) and the discrimination scores on generated samples (reflecting sample quality). By incorporating both aspects, GAM provides a comprehensive evaluation of GAN models.

- **Limitations:** While GAM provides a quantitative metric for comparing GAN models, it has some limitations. It assumes that the performance of the discriminator reflects the overall quality of the GAN model, which may not always be the case. Additionally, GAM's evaluation may be affected by the choice of discriminators, and the metric does not capture all aspects of GAN performance, such as mode collapse or convergence stability.

**History:** The Generative Adversarial Metric (GAM) was introduced in a workshop track at the International Conference on Learning Representations (ICLR) in 2016. The authors proposed GAM as an alternative evaluation metric for GANs, aiming to overcome the limitations of existing evaluation methods. The metric was presented in the workshop paper titled "Generative Adversarial Metric" by Im, Memisevic, Kim, and Jiang. The paper outlined the concept and formulation of GAM, highlighting its potential benefits in evaluating GAN models.

**Related Links**: 
- [Generating images with recurrent adversarial networks](https://arxiv.org/abs/1602.05110)


# Geometry Score

- **Year Introduced:** 2018

- **Basic Idea:** The Geometry Score is a novel metric introduced in 2018 for assessing the quality and diversity of generated samples in generative adversarial networks (GANs). It utilizes the machinery of topology to compare the geometrical properties of the underlying data manifold and the generated samples. By analyzing the topological properties, such as loops and higher-dimensional holes, it provides both qualitative and quantitative means for evaluating GAN performance.

- **Usage:** The Geometry Score can be applied to datasets of various natures, not limited to visual data, making it a versatile metric for evaluating GANs. It allows researchers to gain insights into the properties of GANs, including the quality of generated samples and the presence of mode collapse.

- **Advantages:** The Geometry Score offers a unique approach to GAN evaluation by leveraging topological properties. It takes into account the complexity and non-linear structure of the data manifold and compares it to the generated manifold. By focusing on topological features, it provides a different perspective on GAN performance and can capture aspects such as mode collapse and diversity.

- **Limitations:** The Geometry Score requires approximating the underlying manifolds based on samples, which introduces some level of approximation error. The metric relies on choosing appropriate landmarks and witness points, which can be challenging and impact the results. Additionally, the Geometry Score may not fully capture all aspects of GAN performance, such as semantic coherence or perceptual quality.

**Related Links**: 
- [Geometry Score: A Method For Comparing Generative Adversarial Networks](https://arxiv.org/abs/1802.02664)

# Image Gradients
- **Basic Idea:** The gradient of an image measures how it is changing. It provides information about the magnitude and direction of the image's rapid changes. The gradient is represented as a vector, where the length of the vector represents the magnitude of the gradient, and the direction represents the direction of the most rapid change. Computing the gradient involves combining the partial derivatives of the image in the x and y directions.
- **Usage:** Image gradients are commonly used in computer vision and image processing tasks, such as edge detection, boundary detection, and feature extraction. By analyzing the changes in intensity and direction of the gradients, we can identify edges and boundaries between objects in an image.
- **Advantages:** 
  - Provides information about the magnitude and direction of rapid changes in an image.
  - Useful for edge detection, as image gradients tend to be high at object boundaries.
  - Allows for the extraction of features from images by analyzing the gradients.
- **Limitations:** 
  - Assumes that the derivative of the image doesn't change significantly when moving a small amount in the image.
  - Limited to detecting changes in intensity and direction and may not capture other important image attributes.
  - Can be sensitive to noise and variations in image quality.

# Image Retrieval Performance
 - **Basic Idea:** Image retrieval performance can be evaluated using various metrics depending on the specific task and requirements. Some commonly used metrics for image retrieval evaluation include precision, recall, mean average precision (mAP), normalized discounted cumulative gain (nDCG), and precision at K (P@K). These metrics assess the effectiveness of an image retrieval system in retrieving relevant images based on a query.


# Inception Score (IS)
- **Year Introduced:** 2016
- **Basic Idea:** Inception Score (IS) is a metric used to evaluate the quality and diversity of generated images in the context of generative adversarial networks (GANs). It leverages a pre-trained Inception network to measure the conditional label distribution of generated images and assess their quality based on classification performance.
- **Usage:** IS can be used to compare different GAN models or variations of the same model. It provides a quantitative measure to assess the performance of generative models, aiding researchers and practitioners in model evaluation and selection.
- **Advantages:**
  - Quality Assessment: IS indirectly evaluates the visual quality and realism of generated samples by assessing their classification performance using a pre-trained Inception network.
  - Diversity Evaluation: IS measures the diversity of generated images by computing the entropy of the conditional label distribution, indicating the variety among the samples.
  - Easy Implementation: IS can be implemented by passing generated images through a pre-trained Inception network and analyzing the conditional label distribution, making it accessible for researchers and practitioners.
- **Limitations:**
  - Dependency on Pre-trained Inception Network: The quality of IS is dependent on the performance and generalization ability of the pre-trained Inception network. Different versions of the network or alternative pre-trained classifiers may yield different evaluation results.
  - Limited Scope: IS focuses primarily on quality and diversity aspects, overlooking semantic consistency, fine-grained details, and perceptual similarity to real images.
  - Lack of Human Perception: IS solely relies on the classification performance of the Inception network and does not directly incorporate human perception or subjective evaluations.

**Related Links**: 
[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
[A Note on the Inception Score](https://arxiv.org/abs/1801.01973)

# Kernel Inception Distance (KID)
- **Year Introduced:** 2018
- **Basic Idea:**
  The Kernel Inception Distance (KID) is a measure of the distance between the distribution of Inception features (activations from one of the last layers of the Inception model) of real and generated images. It provides a method for assessing the quality of images generated by Generative Adversarial Networks (GANs).

  The theory behind KID is based on the Maximum Mean Discrepancy (MMD), a measure of the distance between two probability distributions. KID uses the MMD in a Reproducing Kernel Hilbert Space (RKHS), with the Inception features as the input data and a Gaussian kernel as the reproducing kernel.

  The squared MMD is estimated using unbiased estimators based on the pairwise distances between the Inception features of real and generated images.

  The formula for the MMD-based KID is as follows:

  $$\text{KID} = \frac{1}{m} \cdot || \mu_r - \mu_g ||² + \frac{1}{m} \cdot (\text{Var}_r + \text{Var}_g - 2\cdot\text{Cov}_{rg})$$

  where:

  - $m$ is the number of samples,
  - $\mu_r$ and $\mu_g$ are the means of the Inception features of the real and generated images, respectively,
  - $\text{Var}_r$ and $\text{Var}_g$ are the variances of the Inception features of the real and generated images, respectively, and
  - $\text{Cov}_{rg}$ is the covariance between the Inception features of the real and generated images.

- **Usage:** MMD finds its applications in domain adaptation, where the aim is to adapt a model trained on one distribution (the source domain) to perform well on a different distribution (the target domain). For example, a study proposed incorporating MMD into the loss function of autoencoders to reduce mismatches between different data sources in speaker verification systems.


- **Advantages:** KID provides a more reliable measure for evaluating GANs as it does not make any assumptions about the distribution of the samples. It is also relatively straightforward to compute and does not involve complex statistical computations.

- **Limitations:** KID relies on the Inception model, which was trained on the ImageNet dataset. This means it might not be suitable for datasets that are significantly different from ImageNet. It also does not provide a human-interpretable score, which makes it challenging to understand how good a model is by looking at the KID value alone.


**Related Links**: 
- [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)
- [A Study on the Evaluation of Generative Models](https://arxiv.org/abs/2206.10935)


# Maximum Mean Discrepancy (MMD)


- **Basic Idea:**
  Maximum Mean Discrepancy (MMD) is a method used in machine learning to measure the distance between two probability distributions. Introduced as a nonparametric method, MMD doesn't assume any particular form for the distribution of data, making it particularly useful in tasks where machine learning system's performance degrades due to distribution differences in training and test data.

  The fundamental idea of MMD is to provide a measure of similarity between two data sets. If the data sets are drawn from the same distribution, the MMD between them should be small. Conversely, if they're drawn from different distributions, the MMD should be large. Hence, MMD is a valuable tool for tasks such as two-sample testing and training generative models.

- **Formula for Calculation:**
  Empirical estimation of MMD
  In real-life settings, we don't have access to the underlying distribution of our data. Hence, it's possible to use an estimate for MMD as:


  $$MMD^2(P, Q) = \frac{1}{m(m-1)} \sum_{i \neq j}^m k(x_i, x_j) - \frac{2}{mn} \sum_{i,j}^m k(x_i, y_j) + \frac{1}{n(n-1)} \sum_{i \neq j}^n k(y_i, y_j)$$



- **Advantages:** MMD is a flexible and powerful method for comparing distributions, usable with any data type as long as a suitable kernel is available. It has strong theoretical properties: with a properly chosen kernel, MMD can match up to infinite moments of data distributions.

- **Limitations:** However, it requires a suitable choice of kernel and the computation of mean embeddings of the distributions, which can be computationally expensive for large datasets.

**Related Links**: 
- [Minimax Estimation of Maximum Mean Discrepancy with Radial Kernels](https://proceedings.neurips.cc/paper_files/paper/2016/file/5055cbf43fac3f7e2336b27310f0b9ef-Paper.pdf)
- [Maximum Mean Discrepancy (MMD) in Machine Learning](https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html)

# Modified Inception Score (m-IS)
# Multi-Scale Structural Similarity Index Measure (MS-SSIM)
# Normalized Relative Discriminative Score (NRDS)
# Peak Signal-to-Noise Ratio (PSNR)
Precision, Recall, and F1 Score​
# Reconstruction Error
# Relative Average Spectral Error (RASE)
# Root Mean Squared Error Using Sliding Window
# Structural Similarity Index Measure (SSIM) and Multi-Scale SSIM (MS-SSIM)
# The Wasserstein Critic
# Total Variation (TV)
# Universal Image Quality Index
