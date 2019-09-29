### *KGAT: Knowledge Graph Attention Network for Recommendation*

#### 1. Task Formulation

* **1.1. User-Item Bipartite Graph** 

  * Represent interaction data as user-item bipartite graph $\mathcal{G}_1$

    $\mathcal{G}_1=\{(u,y_{ui},i)|u\in\mathcal{U},i\in\mathcal{I}\}$

    where $\mathcal{I}$ and $\mathcal{U}$ separately denote the **user** and **item** sets

* **1.2. Knowledge Graph**: Size information for items

  * The side information is orgnized in the form of knowledge graph $\mathcal{G}_2$.

    $$\mathcal{G}_2=\{(h,r,t)|h,t \in \mathcal{E}, r\in \mathcal{R}\}$$

  * A set of item-entity alignments  is established to indicates that item $i$ can ben aligned with an entity $e$ in the KG.

    $\mathcal{A}=\{(i,e)|i\in \mathcal{I},e\in\mathcal{E}\}$

* **1.3. Collaborative Knowledge Graph**

  * **Definition**: encodes user behaviors and knowledge as a unified relational graph.

    * **User behavior**: $(u, Interact, i)$, where $y_{ui}=1$ represent each user behavior as an additional relation $Interact$ between user $u$ and item $i$

    * **Integrate KG and user-item graph as a unified graph** 

      $$\mathcal{G}=\{(h,r,t)|h,t\in \mathcal{E'}, r\in \mathcal{R'}\}$$

      where $\mathcal{E'}=\mathcal{E}\cup\mathcal{U}$ and $\mathcal{R'}=\mathcal{R}\cup{Interact}$

  * **Task Description**

    * **Input:** $\mathcal{G}$
    * **Output:** A prediction function that predicts the probability $\hat{y}_{ui}$ that user $u$ would adopt item $i$

**2. Methodology**

* **2.1. Embedding Layer**

  * **TransR:**  $$g(h,r,t)=\|W_re_h+e_r-W_re_t\|_2^2$$

  * **Pairwise ranking loss**:

    $$\mathcal{L}_{KG}=\sum\limits_{(h,r,t,t')\in\mathcal{T}}-\ln\sigma(g(h,r,t')-g(h,r,t))$$

* 2.2 **Attentive Embedding Propagtion Layers**

  * **Information Propagation**

    $$e_{\mathcal{N}}=\sum\limits_{(h,r,t)\in\mathcal{N}_h}\pi(h,r,t)e_t$$

    where $\mathcal{N}=\{(h,r,t)\in \mathcal{G}\}$ denotes the set of triplets where $h$ is the head entity.

    $\pi(h,r,t)$ controls the decay factor on each propagation on edge $(h,r,t)$, indication how much information being propagated from $t$ to $h$ conditioned to relation $r$.

  * **Knowledge-aware Attention:** 

    - $$\pi(h,r,t)=(W_r,e_t)^T\tanh(W_re_h+e_r)$$

    - **Normalization:** 

      $$\pi(h,r,t)=\dfrac{\exp(\pi(h,r,t))}{\sum_{(h,r',t')\in\mathcal{N}}\exp(\pi(h,r',t'))}$$

  * **Information Aggregation**

    * **GCN Aggregation**

      $$f_{GCN}=LeakyReLU(W(e_h+e_{\mathcal{N}_h}))$$

    * **GraphSage**

      $$f_{GraphSage}=LeakyReLu(W(e_h||e_{\mathcal{N}_h}))$$, 

      where $||$ stands for concanation operation.

    * **Bi-Interaction**

      $$f_{Bi-Interaction}=LeakyReLU(W_1(e_h+e_{\mathcal{N}_h}))+LeakyReLU(W_2(e_h\odot e_{\mathcal{N}_h}))$$,

      where $\odot$ stands for element-wise product.

  * **High-order Properation** 

    * $$e_h^{(l)}=f(e_h^{(l-1)},e_{\mathcal{N}_h}^{(l-1)})$$

    * $$e_{\mathcal{N}_h}^{(l-1)}=\sum\limits_{(h,r,t)\in\mathcal{N}_h}\pi(h,r,t)e_{t}^{(l-1)}$$

      $e_t^{(l-1)}$ is the presentation of entity $t$ generated from the previous information propagation steps.