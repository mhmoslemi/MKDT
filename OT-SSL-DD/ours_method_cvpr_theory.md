# Self-Supervised Dataset Distillation through Cross-View Prototype Information

Self-supervised dataset distillation seeks a small image set that supports learning useful representations of a much larger unlabeled dataset. Our premise is that such a set should preserve more than the average appearance or feature response of the original data: it should provide reference images through which different visual content remains distinguishable across augmented views. We therefore treat synthetic images as an image-valued dictionary and optimize the information carried by cross-view assignments to this dictionary. The teacher supplies the representation geometry; the synthetic images determine the reference points within that geometry.

Let \(\mathcal D=\{x_i\}_{i=1}^{N}\) denote the unlabeled real dataset and \(\mathcal S=\{s_k\}_{k=1}^{M}\) the synthetic set, with \(M\ll N\). Each \(s_k\in[0,1]^{C\times H\times W}\) is a learnable image. We assume access to a fixed encoder \(f_\theta\) pretrained self-supervised on the real dataset. Its parameters remain unchanged: the dictionary is learned by modifying the images, rather than by changing the feature space in which they are compared. The prototype indices are dictionary entries, not semantic labels, and \(M\) need not equal the number of classes.

For an augmentation \(t\), define the normalized representation

$$
\psi_t(x)=\frac{f_\theta(t(x))}{\|f_\theta(t(x))\|_2}.
$$

Two transformations \(t_u,t_v\) are sampled independently. Each transformation is shared between the real images and the synthetic dictionary, so that a real image and its candidate prototypes are compared under the same viewing condition. For each view \(a\in\{u,v\}\), we assign a real image softly to the synthetic prototypes:

$$
q_a(k\mid x,\mathcal S)=
\frac{\exp\!\left(\psi_{t_a}(x)^\top\psi_{t_a}(s_k)/\tau\right)}
{\sum_{\ell=1}^{M}\exp\!\left(\psi_{t_a}(x)^\top\psi_{t_a}(s_\ell)/\tau\right)},
\qquad \tau>0.
$$

The vector \(q_a(\cdot\mid x,\mathcal S)\) represents the real image relative to the entire synthetic set. It introduces no prescribed correspondence between individual real and synthetic samples. Instead, every prototype competes to explain the image within the teacher's feature geometry. The same prototype is represented by its corresponding transformed image in each view, preserving the connection between the dictionary's feature geometry and its underlying pixels.

These assignments have a soft-quantization interpretation. For unit-norm features \(h\) and \(z_k\), the identity \(\|h-z_k\|_2^2=2-2h^\top z_k\) gives

$$
q(k)=\frac{\exp\!\left(-\|h-z_k\|_2^2/(2\tau)\right)}
{\sum_\ell\exp\!\left(-\|h-z_\ell\|_2^2/(2\tau)\right)}.
$$

Equivalently, \(q\) minimizes \(\sum_k a_k\|h-z_k\|_2^2/2+\tau\sum_k a_k\log a_k\) over probability vectors \(a\). Thus the assignments are entropy-smoothed nearest-prototype codes on the unit sphere. This variational identity explains the assignment rule; it does not introduce an additional term into the distillation objective. Unlike an unconstrained feature dictionary, our dictionary is realized by images, allowing its entries to be transformed and subsequently used as inputs for representation learning.

The central question is whether these codes retain informative relationships across views of the same real images. For a real minibatch \(\mathcal B\) of size \(B\), we aggregate paired assignments into the joint matrix

$$
J_{k\ell}=\frac{1}{B}\sum_{x\in\mathcal B}
q_u(k\mid x,\mathcal S)\,q_v(\ell\mid x,\mathcal S),
\qquad
P=\frac{J+J^\top}{2}.
$$

Each real image contributes the outer product of its two assignment distributions. Pairing is essential: the two factors refer to the same underlying image, not independently chosen real samples. Before symmetrization, \(J\) is the joint distribution obtained by choosing a real image uniformly from the minibatch and independently drawing prototype indices from its two assignment distributions. Symmetrization removes the preference for an ordering of the views. The resulting \(P\) is a nonnegative, normalized joint distribution over pairs of prototype indices. Its marginals are \(p_k^u=\sum_\ell P_{k\ell}\) and \(p_\ell^v=\sum_k P_{k\ell}\).

We learn the synthetic dictionary by maximizing the mutual information of this joint:

$$
\boxed{
\mathcal S^\star=
\arg\min_{\mathcal S\in\mathcal X^M}
\mathcal L(\mathcal S),
\qquad
\mathcal L(\mathcal S)=
-\mathbb E_{\mathcal B,t_u,t_v}
\left[
\sum_{k,\ell}P_{k\ell}
\log\frac{P_{k\ell}}{p_k^u p_\ell^v}
\right],
}
$$

where \(\mathcal X=[0,1]^{C\times H\times W}\), with the usual convention \(0\log 0=0\). This is a single objective. The expectation is over real minibatches and paired transformations; information is computed for each sampled joint before taking the expectation. It is therefore not interchangeable with the information of a joint obtained by pooling all transformations and minibatches.

The information criterion addresses the tension between cross-view predictability and discriminative capacity. Writing \(C_u,C_v\) for the categorical variables with joint \(P\),

$$
I_P(C_u;C_v)
=H_P(C_u)-H_P(C_u\mid C_v)
=D_{\mathrm{KL}}\!\left(P\,\|\,p^u(p^v)^\top\right).
$$

The conditional-entropy interpretation favors predictable relationships between assignments across views, while the marginal-entropy term favors a dictionary capable of expressing multiple outcomes. The KL interpretation measures departure from independent assignments. Both interpretations describe the same loss, rather than separately weighted consistency and diversity penalties. Maximizing information between paired assignments is closely related to the principle used in [Invariant Information Clustering](https://arxiv.org/abs/1807.06653); here, assignments are parameterized by learnable image prototypes under a fixed teacher, rather than by a learned clustering network.

Agreement alone is insufficient: assigning every real image the same distribution in both views makes the assignments agree without distinguishing any content. Marginal entropy alone is also insufficient: an identical uniform assignment for every image has maximal marginal entropy but no cross-view information. Under either identical-constant construction, \(P\) factorizes and \(I_P(C_u;C_v)=0\). In contrast, an ideal deterministic assignment to \(K\) equally populated, corresponding groups has zero conditional entropy and information \(\log K\). More generally,

$$
0\le I_P(C_u;C_v)
\le\min\{H_P(C_u),H_P(C_v)\}
\le\log M.
$$

These bounds follow from nonnegativity of mutual information and conditional entropy, together with the maximum entropy of an \(M\)-valued variable. They explain the dictionary's information capacity, not a guarantee that all prototypes are used or that a larger dictionary produces better representations. The deterministic construction is an ideal reference distribution, not a claim that finite-temperature soft assignments attain it exactly.

Augmentation extends this criterion from a static arrangement of features to relationships observed under changing views. A prototype must participate in informative assignment statistics after transformations are applied to both its pixels and the real images. Sharing the viewing condition controls one source of mismatch between the two sets, while the independently sampled second condition exposes their cross-view relationship. This motivates learning image-valued prototypes rather than merely selecting isolated teacher feature vectors. It does not imply exact invariance to every transformation or preservation of the full real-data distribution.

The information objective should also be distinguished from a requirement of identical prototype indices across views. A joint supported on a consistent permutation can carry as much information as a diagonal joint. Moreover, a symmetrized joint can contain dependence induced by view identity: if assignments are constant across images but differ between the two views, \((ab^\top+ba^\top)/2\) need not factorize. The criterion therefore encourages informative assignment dependence, but does not by itself prove semantic correspondence or exclude every view-dependent shortcut. Its semantic motivation rests on the teacher's learned geometry and on transformations that retain useful visual content.

The resulting connection to dataset distillation is a surrogate principle: an image dictionary that supports distinguishable, predictable teacher-space responses across views may provide useful visual structure for a fresh self-supervised learner. Assignment-based cross-view learning has precedent in [SwAV](https://arxiv.org/abs/2006.09882), while [Distribution Matching dataset condensation](https://arxiv.org/abs/2110.04181) motivates synthesizing compact training sets through feature-space criteria. Our formulation instead places a paired-assignment information objective on the synthetic images themselves. It neither equates that objective with the training loss of a new encoder nor establishes a downstream generalization bound. The theoretical role of the objective is to define and motivate the structure sought in the synthetic dictionary; whether that structure supports representation learning remains an empirical question.
