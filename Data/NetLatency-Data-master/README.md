# Network Latency Datasets

This repository consists datasets of network latencies which we collected from the public network platform `PlanetLab`. Please feel free to contact us if you have any comments or questions.

---

### PlanetLab Dataset

**18 time slices of RTTs between 490 nodes in the PlanetLab network**

This dataset consists of round trip times (RTTs) between 490 nodes in the PlanetLab network over 18 time slices.

The file name of each frame is as the format `PlanetLabData_t`, where `t` is the number of the time slice for this frame. The number `t` ranges from 1 to 18. Each file is a 490x490 matrix, where the (*i*, *j*)-th entry indicates the measured RTT from node *i* to node *j*.


---

## Citation

IF YOU USE THIS PACKAGE IN ANY PUBLISHED RESEARCH, PLEASE KINDLY CITE THE FOLLOWING PAPER:

Rui Zhu, Bang Liu, Di Niu, Zongpeng Li, Hong Vicky Zhao, "Network Latency Estimation for Personal Devices: A Matrix Completion Approach", IEEE/ACM Transactions on Networking, vol. 25, no. , pp. 724-737, April 2017.

## Acknowledgements
- PlanetLab (http://www.planet-lab.org/) for use to collect the dataset

## License
The MIT License (MIT)

