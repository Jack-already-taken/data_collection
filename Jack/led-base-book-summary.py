import sklearn
import torch
from transformers import pipeline

hf_name = "pszemraj/led-base-book-summary"

summarizer = pipeline(
    "summarization",
    hf_name,
    device=0 if torch.cuda.is_available() else -1,
)

wall_of_text = "A typical feed-forward neural field algorithm. Spatiotemporal coordinates are fed into a neural network that predicts values in the reconstructed domain. Then, this domain is mapped to the sensor domain where sensor measurements are available as supervision. Class and Section Problems Addressed Generalization (Section 2) Inverse problems, ill-posed problems, editability; symmetries. Hybrid Representations (Section 3) Computation & memory efficiency, representation capacity, editability: Forward Maps (Section 4) Inverse problems Network Architecture (Section 5) Spectral bias, integration & derivatives. Manipulating Neural Fields (Section 6) Edit ability, constraints, regularization. Table 2: The five classes of techniques in the neural field toolbox each addresses problems that arise in learning, inference, and control. (Section 3). We can supervise reconstruction via differentiable forward maps that transform Or project our domain (e.g, 3D reconstruction via 2D images; Section 4) With appropriate network architecture choices, we can overcome neural network spectral biases (blurriness) and efficiently compute derivatives and integrals (Section 5). Finally, we can manipulate neural fields to add constraints and regularizations, and to achieve editable representations (Section 6). Collectively, these classes constitute a 'toolbox' of techniques to help solve problems with neural fields There are three components in a conditional neural field: (1) An encoder or inference function € that outputs the conditioning latent variable 2 given an observation 0 E(0) =2. 2 is typically a low-dimensional vector, and is often referred to aS a latent code Or feature code_ (2) A mapping function 4 between Z and neural field parameters O: Y(z) = O; (3) The neural field itself $. The encoder € finds the most probable z given the observations O: argmaxz P(2/0). The decoder maximizes the inverse conditional probability to find the most probable 0 given Z: arg- max P(Olz). We discuss different encoding schemes with different optimality guarantees (Section 2.1.1), both global and local conditioning (Section 2.1.2), and different mapping functions Y (Section 2.1.3) 2. Generalization Suppose we wish to estimate a plausible 3D surface shape given a partial or noisy point cloud. We need a suitable prior over the sur- face in its reconstruction domain to generalize to the partial observations. A neural network expresses a prior via the function space of its architecture and parameters 0, and generalization is influenced by the inductive bias of this function space (Section 5)."

result = summarizer(
    wall_of_text,
    min_length=8,
    max_length=256,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    do_sample=False,
    early_stopping=True,
)
print(result[0])

