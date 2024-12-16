# Basic-VAE
Basic implementation of Varational Auto Encoder with pytorch.
Trained for MNIST dataset and CelebA dataset.

# Results

### MNIST
<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <tr>
    <th style="width: 33.33%; text-align: center;">Original Image</th>
    <th style="width: 33.33%; text-align: center;">Reconstructed Image</th>
    <th style="width: 33.34%; text-align: center;">Random Image</th>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="./assets/mnist_original.png" alt="MNIST Original" style="width: 100%; height: auto;">
    </td>
    <td style="text-align: center;">
      <img src="./assets/mnist_reconstruct.png" alt="MNIST Reconstructed" style="width: 100%; height: auto;">
    </td>
    <td style="text-align: center;">
      <img src="./assets/mnist_random.png" alt="MNIST Random" style="width: 100%; height: auto;">
    </td>
  </tr>
</table>

### CelebA
<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <tr>
    <th style="width: 33.33%; text-align: center;">Original Image</th>
    <th style="width: 33.33%; text-align: center;">Reconstructed Image</th>
    <th style="width: 33.34%; text-align: center;">Random Image</th>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="./assets/celeba_original.png" alt="CelebA Original" style="width: 100%; height: auto;">
    </td>
    <td style="text-align: center;">
      <img src="./assets/celeba_reconstruct.png" alt="CelebA Reconstructed" style="width: 100%; height: auto;">
    </td>
    <td style="text-align: center;">
      <img src="./assets/celeba_random.png" alt="CelebA Random" style="width: 100%; height: auto;">
    </td>
  </tr>
</table>
