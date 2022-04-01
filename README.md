# InterFusion
This is a repo of [InterFusion](https://scholar.google.com/) for 3D object detection.

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
<!-- 
![image](https://github.com/Link2Link/FE_GCN/blob/main/fig/full_stureture2.png)
![image](https://github.com/Link2Link/FE_GCN/blob/main/fig/figure_gt_pp_fe.png) -->

## Introduction
Many recent works detect 3D objects by several
sensor modalities for autonomous driving, where high-resolution
cameras and high-line Lidars are mostly used but relatively
expensive. To achieve a balance between overall cost and detection
accuracy, many multi-modal fusion techniques have been suggested.
In recent years, the fusion of Lidar and Radar has gained
ever-increasing attention, especially 4D Radar, which can adapt to
bad weather conditions due to its penetrability. Although features
have been fused from multiple sensing modalities, most methods
cannot learn interactions from different modalities, which does
not make for their best use. Inspired by the self-attention mechanism,
here we present InterFusion, an interaction-based fusion
framework, to fuse 16-line Lidar with 4D Radar. It aggregates
features from two modalities and identifies cross-modal relations
between Radar and Lidar features. In experimental evaluations
on the Astyx HiRes2019 dataset, our method outperformed the
baseline by 4.09% mAP in 3D and 10.34% BEV mAP for
the car class at the moderate level. 
* Model Framework:
<p align="center">
  <img src="docs/framework.png" width="95%">
</p>

## Experiment Results:
* All experiments are tested on Astyx Hires2019
<table>
   <tr>
      <td>Modality </td>
      <td> Method 3D</td>
      <td></td>
      <td></td>
      <td></td>
      <td>mAP(%) BEV mAP(%)</td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td>Easy </td>
      <td>Moderate</td>
      <td>Hard</td>
      <td>Easy </td>
      <td>Moderate</td>
      <td>Hard</td>
   </tr>
   <tr>
      <td>Radar</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>PointRCNN </td>
      <td>12.23</td>
      <td>9.1</td>
      <td>9.1</td>
      <td>14.95</td>
      <td>13.82</td>
      <td>13.89</td>
   </tr>
   <tr>
      <td>SECOND [20] </td>
      <td>24.11</td>
      <td>18.5</td>
      <td>17.77</td>
      <td>41.25</td>
      <td>30.58</td>
      <td>29.33</td>
   </tr>
   <tr>
      <td>PVRCNN [33]</td>
      <td>28.21</td>
      <td>22.29</td>
      <td>20.4</td>
      <td>46.62</td>
      <td>35.1</td>
      <td>33.67</td>
   </tr>
   <tr>
      <td>PointPillars [30] </td>
      <td>30.14</td>
      <td>24.06</td>
      <td>21.91</td>
      <td>45.66</td>
      <td>36.71</td>
      <td>35.3</td>
   </tr>
   <tr>
      <td>Lidar</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>PointRCNN [23] </td>
      <td>30.67</td>
      <td>23.69</td>
      <td>23.03</td>
      <td>35.75</td>
      <td>28.13</td>
      <td>23.79</td>
   </tr>
   <tr>
      <td>SECOND [20] </td>
      <td>53.32</td>
      <td>44.1</td>
      <td>40.16</td>
      <td>57.26</td>
      <td>47.52</td>
      <td>45.4</td>
   </tr>
   <tr>
      <td>PVRCNN [33] </td>
      <td>54.93</td>
      <td>45.29</td>
      <td>41.4</td>
      <td>56.71</td>
      <td>47.55</td>
      <td>45.06</td>
   </tr>
   <tr>
      <td>PointPillars [30] </td>
      <td>53.02</td>
      <td>43.56</td>
      <td>41.72</td>
      <td>55.76</td>
      <td>45.81</td>
      <td>43.62</td>
   </tr>
   <tr>
      <td>Radar+Lidar Ours(InterFusion) </td>
      <td>59.04</td>
      <td>47.65</td>
      <td>46.47</td>
      <td>68.1</td>
      <td>56.15</td>
      <td>55.01</td>
   </tr>
   <tr>
      <td>Delta </td>
      <td>+6.02 </td>
      <td>+4.09</td>
      <td>+4.75 </td>
      <td>+12.34</td>
      <td> +10.34</td>
      <td> +11.39</td>
   </tr>
</table>


## Citation 
If you find this project useful in your research, please consider cite:


```

```

