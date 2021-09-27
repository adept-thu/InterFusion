# InterFusion
Implementation for [InterFusin](https://scholar.google.com/) based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
<!-- 
![image](https://github.com/Link2Link/FE_GCN/blob/main/fig/full_stureture2.png)
![image](https://github.com/Link2Link/FE_GCN/blob/main/fig/figure_gt_pp_fe.png) -->

## Introduction
* Model Framework:
<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

## Experiment Results:
* All experiments are tested on Astyx Hires2019
<!-- |                                             | mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|:-------:|:-------:|:---------:|
| [PointPillar-MultiHead](tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml) | 33.87	| 26.00 | 32.07	| 28.74 | 20.15 | 44.63 | 58.23	 | [model-23M](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing) | 
| [SECOND-MultiHead (CBGS)](tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml) | 31.15 |	25.51 |	26.64 | 26.26 | 20.46 | 50.59 | 62.29 | [model-35M](https://drive.google.com/file/d/1bNzcOnE3u9iooBFMk2xK7HqhdeQ_nwTq/view?usp=sharing) |
 -->
\begin{table*}
    \centering
    \begin{tabular}{cccccccc}
    \toprule
    % \toprule
    \multirow{2}{*}{Modality}&\multirow{2}{*}{Method}&
    \multicolumn{3}{c}{3D mAP(\%)}  & \multicolumn{3}{c}{BEV mAP(\%)} \cr
            \cmidrule(lr){3-5} \cmidrule(lr){6-8}
                    & & Easy & Moderate & Hard & Easy & Moderate & Hard \cr
    \cmidrule(lr){1-8}
    \multirow{4}{*}{RaDAR}
    & PointRCNN\cite{8954080} & 12.23 & 9.10 & 9.10 & 14.95 & 13.82 & 13.89 \\
    & SECOND\cite{SECOND} & 24.11 &18.50 &17.77 & 41.25 & 30.58 & 29.33 \\
	& PVRCNN\cite{9157234} & 28.21 & 22.29 & 20.40 & 46.62 & 35.10 & 33.67 \\
	& PointPillars\cite{8954311} & 30.14 & 24.06 & 21.91 & 45.66 & 36.71 & 35.30 \\
    \cmidrule(lr){1-8}
    \multirow{4}{*}{LiDAR}
    & PointRCNN\cite{8954080} & 30.67 & 23.69 & 23.03 & 35.75 & 28.13 & 23.79 \\
	& SECOND\cite{SECOND} & 53.32 & 44.10 & 40.16 & 57.26 & 47.52 & 45.40 \\
	& PointPillars\cite{8954311} & 53.02 & 43.56 & 41.72 & 55.76 & 45.81 & 43.62 \\
	& PVRCNN\cite{9157234} & 54.93 & 45.29 & 41.40 & 56.71 & 47.55 & 45.06 \\
	
	\cmidrule(lr){1-8}
	\multirow{1}{*}{RaDAR+LiDAR} 
	& \textbf{Ours(InterFusion)} & \textbf{59.04} & \textbf{47.65} & \textbf{46.47} & \textbf{68.10} & \textbf{56.15} & \textbf{55.01} \\
	
    \bottomrule
    \end{tabular}\vspace{0cm}
    \caption{Comparative results of current mainstream frameworks on RaDAR and LiDAR.}
    \label{tab:comparative}
\end{table*}

## Citation 
If you find this project useful in your research, please consider cite:


```

```

