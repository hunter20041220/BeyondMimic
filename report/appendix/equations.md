# Equations and Implementation Notes

## Action to PD target

$$
\theta^{sp} = \theta^0 + \alpha \odot a
$$

MuJoCo diagnostic controller:

$$
\tau \approx K_p(\theta^{sp} - \theta) - K_d \dot{\theta}
$$

## VAE objective

$$
\mathcal{L}_{VAE} = \|a - D(o,z)\|_2^2 + \beta D_{KL}(q_\phi(z|o,a)\|N(0,I))
$$

## Diffusion noising

$$
x_k = \sqrt{\bar{\alpha}_k}x_0 + \sqrt{1-\bar{\alpha}_k}\epsilon
$$

## Denoising loss

$$
\mathcal{L}_{diff} = \|\hat{x}_0 - x_0\|_2^2
$$

## Guidance

At test time, task cost gradients modify reverse diffusion samples:

$$
x \leftarrow x - \lambda \nabla_x C(x)
$$

This report treats current guidance as offline/local proxy evidence unless a physical closed-loop rollout exists.
