
<h1 align="center">
Codesign of Humanoid Robots for Ergonomic Collaboration with Multiple Humans via Genetic Algorithms and Nonlinear Optimization 
</h1>


<div align="center">

C.Sartore, L.Rapett, F.Bergonti, S.Dafarra, S.Traversaro, D.Pucci  _"Codesign of Humanoid Robots for Ergonomic Collaboration with Multiple Humans via Genetic Algorithms and Nonlinear Optimization "_
in 2023 IEEE-RAS International Conference on Humanoid Robotics (Humanoids)

</div>

<p align="center">


[![Video](https://github.com/ami-iit/paper_sartore_2022_humanoids_ergonomic_design/assets/56030908/a0d66262-5539-481e-ac42-60219561b607)](https://github.com/ami-iit/paper_sartore_2022_humanoids_ergonomic_design/assets/56030908/6f73779e-5153-4048-bb1d-706c59b80490)

</p>

<div align="center">
  IEEE-RAS International Conference on Humanoid Robotics
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a>
</div>

### Installation


:warning: The repository depends on [HSL for IPOPT (Coin-HSL)](https://www.hsl.rl.ac.uk/ipopt/), to correctly link the library please substitute [this](https://github.com/ami-iit/paper_sartore_2022_humanoids_ergonomic_design/blob/fc5083ca619d9c0dfe4e333fadad6d0f000c0dbf/Dockerfile#L26) line of the docker image with the absolute path to the `coinhsl.zip`. In particular for the paper experiments Coin-HSL 2019.05.21 have been used, but also later version should work fine. 

⚠️ This repository depends on [docker](https://docs.docker.com/)


To install the repo on a Linux terminal follow the following steps 

```
git clone https://github.com/ami-iit/paper_sartore_2023_IROS_Codesign_GA_NL.git
cd paper_sartore_2023_IROS_Codesign_GA_NL
docker build --tag sartore2023results . 
```

### Running 
- `GA_with_PyGad.py` : script to launch the bilevel optimization framework; 
- `GA_processOutput.py` : script to process the output of the genetic algorithm; 
- `CompareTorqueSimulation.py`: script to process the simulation output. 

### Maintainer

This repository is maintained by:

| | |
|:---:|:---:|
| [<img src="https://github.com/CarlottaSartore.png" width="40">](https://github.com/GitHubUserName) | [@CarlottaSartore](https://github.com/CarlottaSartore) |
