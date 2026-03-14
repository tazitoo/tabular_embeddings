#!/usr/bin/env python3
"""Plot complexity (expansion * topk) vs loss for round 8 SAE sweeps."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Data from Optuna studies (round 8, dead penalty objective)
data = {
    "TabICL": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.3407, "total_loss": 0.3692, "alive_pct": 55.2, "stability": 0.888},
        {"expansion": 16, "topk": 16, "recon_loss": 0.4188, "total_loss": 0.4362, "alive_pct": 38.0, "stability": 0.818},
        {"expansion": 4, "topk": 16, "recon_loss": 0.5163, "total_loss": 0.5475, "alive_pct": 83.7, "stability": 0.860},
        {"expansion": 4, "topk": 16, "recon_loss": 0.5317, "total_loss": 0.5600, "alive_pct": 74.5, "stability": 0.813},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0727, "total_loss": 0.1039, "alive_pct": 81.1, "stability": 0.769},
        {"expansion": 8, "topk": 16, "recon_loss": 0.4609, "total_loss": 0.4851, "alive_pct": 72.1, "stability": 0.885},
        {"expansion": 16, "topk": 32, "recon_loss": 0.3764, "total_loss": 0.4039, "alive_pct": 29.6, "stability": 0.875},
        {"expansion": 16, "topk": 64, "recon_loss": 0.2755, "total_loss": 0.3019, "alive_pct": 70.9, "stability": 0.766},
        {"expansion": 8, "topk": 64, "recon_loss": 0.2605, "total_loss": 0.2902, "alive_pct": 79.9, "stability": 0.825},
        {"expansion": 16, "topk": 16, "recon_loss": 0.4999, "total_loss": 0.5309, "alive_pct": 12.9, "stability": 0.893},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0537, "total_loss": 0.0849, "alive_pct": 53.7, "stability": 0.806},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0532, "total_loss": 0.0844, "alive_pct": 54.1, "stability": 0.806},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0587, "total_loss": 0.0899, "alive_pct": 64.1, "stability": 0.806},
        {"expansion": 16, "topk": 128, "recon_loss": 0.1404, "total_loss": 0.1698, "alive_pct": 48.4, "stability": 0.824},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0537, "total_loss": 0.0850, "alive_pct": 28.1, "stability": 0.806},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0976, "total_loss": 0.1288, "alive_pct": 95.7, "stability": 0.661},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0969, "total_loss": 0.1281, "alive_pct": 95.8, "stability": 0.660},
        {"expansion": 4, "topk": 32, "recon_loss": 0.4003, "total_loss": 0.4308, "alive_pct": 89.6, "stability": 0.750},
        {"expansion": 4, "topk": 128, "recon_loss": 0.2225, "total_loss": 0.2537, "alive_pct": 99.2, "stability": 0.649},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0943, "total_loss": 0.1255, "alive_pct": 88.6, "stability": 0.776},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1049, "total_loss": 0.1361, "alive_pct": 92.7, "stability": 0.722},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1055, "total_loss": 0.1368, "alive_pct": 91.4, "stability": 0.723},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1022, "total_loss": 0.1334, "alive_pct": 67.3, "stability": 0.749},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0896, "total_loss": 0.1208, "alive_pct": 94.7, "stability": 0.658},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0887, "total_loss": 0.1199, "alive_pct": 94.3, "stability": 0.659},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0922, "total_loss": 0.1235, "alive_pct": 76.3, "stability": 0.785},
        {"expansion": 4, "topk": 32, "recon_loss": 0.4036, "total_loss": 0.4342, "alive_pct": 91.7, "stability": 0.789},
        {"expansion": 4, "topk": 128, "recon_loss": 0.1619, "total_loss": 0.1932, "alive_pct": 90.2, "stability": 0.772},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0921, "total_loss": 0.1233, "alive_pct": 94.0, "stability": 0.671},
        {"expansion": 8, "topk": 64, "recon_loss": 0.2943, "total_loss": 0.3213, "alive_pct": 37.4, "stability": 0.818},
    ],
    "TabPFN": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.2024, "total_loss": 0.2336, "alive_pct": 93.8, "stability": 0.913},
        {"expansion": 16, "topk": 16, "recon_loss": 0.2664, "total_loss": 0.2898, "alive_pct": 67.4, "stability": 0.857},
        {"expansion": 4, "topk": 16, "recon_loss": 0.3474, "total_loss": 0.3786, "alive_pct": 90.1, "stability": 0.751},
        {"expansion": 4, "topk": 16, "recon_loss": 0.3540, "total_loss": 0.3840, "alive_pct": 87.4, "stability": 0.779},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0214, "total_loss": 0.0526, "alive_pct": 82.6, "stability": 0.758},
        {"expansion": 8, "topk": 16, "recon_loss": 0.2908, "total_loss": 0.3197, "alive_pct": 86.5, "stability": 0.799},
        {"expansion": 16, "topk": 32, "recon_loss": 0.1672, "total_loss": 0.1972, "alive_pct": 37.6, "stability": 0.917},
        {"expansion": 16, "topk": 64, "recon_loss": 0.1502, "total_loss": 0.1809, "alive_pct": 89.6, "stability": 0.750},
        {"expansion": 8, "topk": 64, "recon_loss": 0.1310, "total_loss": 0.1621, "alive_pct": 90.2, "stability": 0.765},
        {"expansion": 16, "topk": 16, "recon_loss": 0.2632, "total_loss": 0.2940, "alive_pct": 40.4, "stability": 0.908},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0135, "total_loss": 0.0448, "alive_pct": 41.9, "stability": 0.817},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0526, "total_loss": 0.0839, "alive_pct": 80.7, "stability": 0.757},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0579, "total_loss": 0.0892, "alive_pct": 85.3, "stability": 0.761},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0563, "total_loss": 0.0875, "alive_pct": 74.5, "stability": 0.767},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0748, "total_loss": 0.1061, "alive_pct": 93.1, "stability": 0.678},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0710, "total_loss": 0.1022, "alive_pct": 80.7, "stability": 0.736},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0752, "total_loss": 0.1065, "alive_pct": 96.7, "stability": 0.667},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0758, "total_loss": 0.1070, "alive_pct": 97.9, "stability": 0.664},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1235, "total_loss": 0.1547, "alive_pct": 99.5, "stability": 0.760},
        {"expansion": 4, "topk": 32, "recon_loss": 0.2265, "total_loss": 0.2575, "alive_pct": 89.6, "stability": 0.817},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0868, "total_loss": 0.1181, "alive_pct": 97.0, "stability": 0.704},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0868, "total_loss": 0.1181, "alive_pct": 96.6, "stability": 0.710},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0822, "total_loss": 0.1135, "alive_pct": 95.8, "stability": 0.707},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0917, "total_loss": 0.1229, "alive_pct": 96.1, "stability": 0.723},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0700, "total_loss": 0.1013, "alive_pct": 92.1, "stability": 0.691},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0899, "total_loss": 0.1212, "alive_pct": 98.6, "stability": 0.673},
        {"expansion": 4, "topk": 32, "recon_loss": 0.2838, "total_loss": 0.3150, "alive_pct": 97.8, "stability": 0.702},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1184, "total_loss": 0.1184, "alive_pct": 100.0, "stability": 0.652},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1183, "total_loss": 0.1496, "alive_pct": 100.0, "stability": 0.711},
        {"expansion": 4, "topk": 64, "recon_loss": 0.1735, "total_loss": 0.2048, "alive_pct": 90.4, "stability": 0.763},
    ],
    "Mitra": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.1192, "total_loss": 0.1497, "alive_pct": 77.6, "stability": 0.920},
        {"expansion": 16, "topk": 16, "recon_loss": 0.1494, "total_loss": 0.1725, "alive_pct": 26.9, "stability": 0.843},
        {"expansion": 4, "topk": 16, "recon_loss": 0.1999, "total_loss": 0.2311, "alive_pct": 58.0, "stability": 0.852},
        {"expansion": 4, "topk": 16, "recon_loss": 0.2007, "total_loss": 0.2287, "alive_pct": 66.7, "stability": 0.859},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0572, "total_loss": 0.0884, "alive_pct": 89.9, "stability": 0.818},
        {"expansion": 8, "topk": 16, "recon_loss": 0.1654, "total_loss": 0.1887, "alive_pct": 57.6, "stability": 0.898},
        {"expansion": 16, "topk": 32, "recon_loss": 0.1055, "total_loss": 0.1368, "alive_pct": 14.3, "stability": 0.883},
        {"expansion": 16, "topk": 64, "recon_loss": 0.1026, "total_loss": 0.1316, "alive_pct": 66.2, "stability": 0.809},
        {"expansion": 8, "topk": 64, "recon_loss": 0.0925, "total_loss": 0.1235, "alive_pct": 89.0, "stability": 0.865},
        {"expansion": 16, "topk": 16, "recon_loss": 0.1644, "total_loss": 0.1957, "alive_pct": 6.7, "stability": 0.907},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0491, "total_loss": 0.0803, "alive_pct": 47.1, "stability": 0.847},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0660, "total_loss": 0.0972, "alive_pct": 91.6, "stability": 0.862},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0693, "total_loss": 0.1005, "alive_pct": 89.0, "stability": 0.861},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0708, "total_loss": 0.1020, "alive_pct": 93.5, "stability": 0.860},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0466, "total_loss": 0.0779, "alive_pct": 48.2, "stability": 0.812},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0575, "total_loss": 0.0887, "alive_pct": 41.3, "stability": 0.837},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0613, "total_loss": 0.0926, "alive_pct": 65.5, "stability": 0.882},
        {"expansion": 8, "topk": 32, "recon_loss": 0.1205, "total_loss": 0.1500, "alive_pct": 69.9, "stability": 0.867},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0781, "total_loss": 0.1094, "alive_pct": 85.1, "stability": 0.849},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0886, "total_loss": 0.1199, "alive_pct": 98.9, "stability": 0.850},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0767, "total_loss": 0.1079, "alive_pct": 95.8, "stability": 0.900},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0764, "total_loss": 0.1077, "alive_pct": 95.2, "stability": 0.905},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0715, "total_loss": 0.1028, "alive_pct": 91.0, "stability": 0.908},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0690, "total_loss": 0.1002, "alive_pct": 77.1, "stability": 0.912},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0974, "total_loss": 0.1287, "alive_pct": 97.6, "stability": 0.909},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0633, "total_loss": 0.0946, "alive_pct": 58.2, "stability": 0.896},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0915, "total_loss": 0.1228, "alive_pct": 99.1, "stability": 0.883},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0996, "total_loss": 0.1105, "alive_pct": 99.4, "stability": 0.886},
        {"expansion": 4, "topk": 32, "recon_loss": 0.1495, "total_loss": 0.1797, "alive_pct": 86.5, "stability": 0.895},
        {"expansion": 4, "topk": 128, "recon_loss": 0.1206, "total_loss": 0.1376, "alive_pct": 99.5, "stability": 0.835},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0989, "total_loss": 0.1301, "alive_pct": 99.4, "stability": 0.886},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0851, "total_loss": 0.1163, "alive_pct": 98.7, "stability": 0.910},
        {"expansion": 4, "topk": 128, "recon_loss": 0.0869, "total_loss": 0.1181, "alive_pct": 98.6, "stability": 0.909},
        {"expansion": 4, "topk": 128, "recon_loss": 0.1276, "total_loss": 0.1418, "alive_pct": 99.7, "stability": 0.929},
    ],
    "CARTE": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.0678, "total_loss": 0.0989, "alive_pct": 88.2, "stability": 0.925},
        {"expansion": 16, "topk": 16, "recon_loss": 0.0890, "total_loss": 0.1104, "alive_pct": 35.6, "stability": 0.862},
        {"expansion": 4, "topk": 16, "recon_loss": 0.1194, "total_loss": 0.1507, "alive_pct": 29.1, "stability": 0.825},
        {"expansion": 4, "topk": 16, "recon_loss": 0.1211, "total_loss": 0.1473, "alive_pct": 66.7, "stability": 0.835},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0272, "total_loss": 0.0585, "alive_pct": 94.2, "stability": 0.841},
        {"expansion": 8, "topk": 16, "recon_loss": 0.1032, "total_loss": 0.1251, "alive_pct": 59.4, "stability": 0.858},
        {"expansion": 16, "topk": 32, "recon_loss": 0.0443, "total_loss": 0.0755, "alive_pct": 10.9, "stability": 0.906},
        {"expansion": 16, "topk": 64, "recon_loss": 0.0571, "total_loss": 0.0875, "alive_pct": 82.5, "stability": 0.841},
        {"expansion": 8, "topk": 64, "recon_loss": 0.0478, "total_loss": 0.0789, "alive_pct": 89.1, "stability": 0.844},
        {"expansion": 16, "topk": 16, "recon_loss": 0.0866, "total_loss": 0.1178, "alive_pct": 5.6, "stability": 0.929},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0225, "total_loss": 0.0538, "alive_pct": 53.5, "stability": 0.869},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0339, "total_loss": 0.0652, "alive_pct": 90.3, "stability": 0.839},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0359, "total_loss": 0.0671, "alive_pct": 86.4, "stability": 0.837},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0361, "total_loss": 0.0674, "alive_pct": 90.5, "stability": 0.840},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0274, "total_loss": 0.0587, "alive_pct": 41.6, "stability": 0.784},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0214, "total_loss": 0.0527, "alive_pct": 35.6, "stability": 0.880},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0298, "total_loss": 0.0611, "alive_pct": 55.6, "stability": 0.820},
        {"expansion": 8, "topk": 32, "recon_loss": 0.0641, "total_loss": 0.0935, "alive_pct": 72.0, "stability": 0.875},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0390, "total_loss": 0.0702, "alive_pct": 98.6, "stability": 0.882},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0388, "total_loss": 0.0701, "alive_pct": 98.4, "stability": 0.878},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0392, "total_loss": 0.0704, "alive_pct": 98.6, "stability": 0.879},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0378, "total_loss": 0.0691, "alive_pct": 98.5, "stability": 0.878},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0394, "total_loss": 0.0707, "alive_pct": 98.7, "stability": 0.882},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0395, "total_loss": 0.0707, "alive_pct": 99.0, "stability": 0.882},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0392, "total_loss": 0.0704, "alive_pct": 97.8, "stability": 0.893},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0319, "total_loss": 0.0631, "alive_pct": 91.8, "stability": 0.885},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0344, "total_loss": 0.0657, "alive_pct": 97.6, "stability": 0.882},
        {"expansion": 4, "topk": 256, "recon_loss": 0.0487, "total_loss": 0.0487, "alive_pct": 100.0, "stability": 0.863},
        {"expansion": 4, "topk": 32, "recon_loss": 0.0834, "total_loss": 0.1135, "alive_pct": 77.6, "stability": 0.887},
        {"expansion": 4, "topk": 64, "recon_loss": 0.0677, "total_loss": 0.0989, "alive_pct": 94.1, "stability": 0.887},
    ],
    "HyperFast": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.1429, "total_loss": 0.2092, "alive_pct": 93.9, "stability": 0.852},
        {"expansion": 16, "topk": 16, "recon_loss": 0.1355, "total_loss": 0.4616, "alive_pct": 40.5, "stability": 0.815},
        {"expansion": 4, "topk": 16, "recon_loss": 0.2849, "total_loss": 0.5652, "alive_pct": 50.7, "stability": 0.897},
        {"expansion": 4, "topk": 16, "recon_loss": 0.2467, "total_loss": 0.3665, "alive_pct": 82.8, "stability": 0.823},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0759, "total_loss": 0.1343, "alive_pct": 95.5, "stability": 0.792},
        {"expansion": 8, "topk": 16, "recon_loss": 0.1743, "total_loss": 0.3512, "alive_pct": 70.8, "stability": 0.856},
        {"expansion": 16, "topk": 32, "recon_loss": 0.1148, "total_loss": 0.5867, "alive_pct": 12.0, "stability": 0.868},
        {"expansion": 16, "topk": 64, "recon_loss": 0.1235, "total_loss": 0.2334, "alive_pct": 84.9, "stability": 0.780},
        {"expansion": 8, "topk": 64, "recon_loss": 0.1198, "total_loss": 0.1677, "alive_pct": 97.7, "stability": 0.823},
        {"expansion": 16, "topk": 16, "recon_loss": 0.1770, "total_loss": 0.6846, "alive_pct": 4.8, "stability": 0.893},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0771, "total_loss": 0.2629, "alive_pct": 69.8, "stability": 0.818},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1005, "total_loss": 0.1408, "alive_pct": 99.0, "stability": 0.843},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1078, "total_loss": 0.1454, "alive_pct": 98.4, "stability": 0.841},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1067, "total_loss": 0.1335, "alive_pct": 99.3, "stability": 0.836},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1198, "total_loss": 0.2379, "alive_pct": 83.5, "stability": 0.874},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0834, "total_loss": 0.3123, "alive_pct": 61.1, "stability": 0.809},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1071, "total_loss": 0.1922, "alive_pct": 90.1, "stability": 0.889},
        {"expansion": 8, "topk": 32, "recon_loss": 0.1453, "total_loss": 0.2432, "alive_pct": 87.5, "stability": 0.817},
        {"expansion": 16, "topk": 256, "recon_loss": 0.1098, "total_loss": 0.1707, "alive_pct": 95.1, "stability": 0.817},
        {"expansion": 4, "topk": 128, "recon_loss": 0.1563, "total_loss": 0.1786, "alive_pct": 100.0, "stability": 0.838},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0760, "total_loss": 0.1657, "alive_pct": 89.2, "stability": 0.827},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1037, "total_loss": 0.1329, "alive_pct": 99.4, "stability": 0.841},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1048, "total_loss": 0.1391, "alive_pct": 99.8, "stability": 0.835},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1071, "total_loss": 0.1602, "alive_pct": 96.6, "stability": 0.889},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1101, "total_loss": 0.1360, "alive_pct": 100.0, "stability": 0.819},
        {"expansion": 8, "topk": 128, "recon_loss": 0.0967, "total_loss": 0.2019, "alive_pct": 86.1, "stability": 0.867},
        {"expansion": 8, "topk": 32, "recon_loss": 0.1331, "total_loss": 0.2427, "alive_pct": 85.2, "stability": 0.846},
        {"expansion": 8, "topk": 256, "recon_loss": 0.1069, "total_loss": 0.1311, "alive_pct": 100.0, "stability": 0.805},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1023, "total_loss": 0.1938, "alive_pct": 88.9, "stability": 0.842},
        {"expansion": 8, "topk": 64, "recon_loss": 0.1281, "total_loss": 0.2382, "alive_pct": 85.1, "stability": 0.842},
    ],
    "TabDPT": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.2934, "total_loss": 0.5819, "alive_pct": 50.6, "stability": 0.912},
        {"expansion": 16, "topk": 16, "recon_loss": 0.3296, "total_loss": 0.7247, "alive_pct": 26.0, "stability": 0.834},
        {"expansion": 4, "topk": 16, "recon_loss": 0.4336, "total_loss": 0.6838, "alive_pct": 59.5, "stability": 0.901},
        {"expansion": 4, "topk": 16, "recon_loss": 0.4428, "total_loss": 0.656, "alive_pct": 66.3, "stability": 0.88},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0992, "total_loss": 0.2259, "alive_pct": 85.6, "stability": 0.803},
        {"expansion": 8, "topk": 16, "recon_loss": 0.3696, "total_loss": 0.6395, "alive_pct": 52.4, "stability": 0.915},
        {"expansion": 16, "topk": 32, "recon_loss": 0.3291, "total_loss": 0.7591, "alive_pct": 21.3, "stability": 0.88},
        {"expansion": 16, "topk": 64, "recon_loss": 0.2375, "total_loss": 0.529, "alive_pct": 48.9, "stability": 0.803},
        {"expansion": 8, "topk": 64, "recon_loss": 0.2357, "total_loss": 0.4076, "alive_pct": 75.5, "stability": 0.891},
        {"expansion": 16, "topk": 16, "recon_loss": 0.4268, "total_loss": 0.9331, "alive_pct": 5.3, "stability": 0.908},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0812, "total_loss": 0.4749, "alive_pct": 29.1, "stability": 0.823},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1564, "total_loss": 0.273, "alive_pct": 87.7, "stability": 0.876},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1676, "total_loss": 0.2785, "alive_pct": 88.9, "stability": 0.873},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1664, "total_loss": 0.2687, "alive_pct": 90.7, "stability": 0.869},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1085, "total_loss": 0.4735, "alive_pct": 35.2, "stability": 0.843},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0791, "total_loss": 0.5364, "alive_pct": 15.6, "stability": 0.812},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1586, "total_loss": 0.4107, "alive_pct": 59.1, "stability": 0.897},
        {"expansion": 8, "topk": 32, "recon_loss": 0.304, "total_loss": 0.5151, "alive_pct": 66.8, "stability": 0.891},
        {"expansion": 16, "topk": 256, "recon_loss": 0.1732, "total_loss": 0.4102, "alive_pct": 62.1, "stability": 0.826},
        {"expansion": 4, "topk": 128, "recon_loss": 0.1843, "total_loss": 0.2792, "alive_pct": 92.3, "stability": 0.855},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0972, "total_loss": 0.3152, "alive_pct": 66.3, "stability": 0.843},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1608, "total_loss": 0.2705, "alive_pct": 89.2, "stability": 0.873},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1637, "total_loss": 0.2684, "alive_pct": 90.2, "stability": 0.867},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1592, "total_loss": 0.3215, "alive_pct": 78.0, "stability": 0.9},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1785, "total_loss": 0.2756, "alive_pct": 91.8, "stability": 0.852},
        {"expansion": 8, "topk": 128, "recon_loss": 0.158, "total_loss": 0.4501, "alive_pct": 50.6, "stability": 0.885},
        {"expansion": 8, "topk": 32, "recon_loss": 0.2987, "total_loss": 0.4777, "alive_pct": 74.1, "stability": 0.906},
        {"expansion": 8, "topk": 256, "recon_loss": 0.0831, "total_loss": 0.2341, "alive_pct": 80.4, "stability": 0.819},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1431, "total_loss": 0.2277, "alive_pct": 94.5, "stability": 0.848},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1601, "total_loss": 0.2195, "alive_pct": 99.8, "stability": 0.82},
    ],
    "Tabula-8B": [
        {"expansion": 2, "topk": 64, "recon_loss": 0.4537, "total_loss": 0.8597, "alive_pct": 25.0, "stability": 0.892},
        {"expansion": 4, "topk": 16, "recon_loss": 0.3293, "total_loss": 0.7333, "alive_pct": 25.1, "stability": 0.832},
        {"expansion": 1, "topk": 16, "recon_loss": 0.4307, "total_loss": 0.8974, "alive_pct": 12.9, "stability": 0.871},
        {"expansion": 1, "topk": 16, "recon_loss": 0.4279, "total_loss": 0.7691, "alive_pct": 37.5, "stability": 0.846},
        {"expansion": 4, "topk": 256, "recon_loss": 0.2251, "total_loss": 0.5002, "alive_pct": 51.2, "stability": 0.821},
        {"expansion": 2, "topk": 16, "recon_loss": 0.3298, "total_loss": 0.7058, "alive_pct": 30.3, "stability": 0.849},
        {"expansion": 4, "topk": 32, "recon_loss": 0.4419, "total_loss": 0.962, "alive_pct": 2.2, "stability": 0.933},
        {"expansion": 4, "topk": 64, "recon_loss": 0.281, "total_loss": 0.6311, "alive_pct": 35.8, "stability": 0.816},
        {"expansion": 2, "topk": 64, "recon_loss": 0.2722, "total_loss": 0.4609, "alive_pct": 68.5, "stability": 0.837},
        {"expansion": 4, "topk": 16, "recon_loss": 0.5574, "total_loss": 1.0833, "alive_pct": 1.1, "stability": 0.943},
        {"expansion": 2, "topk": 128, "recon_loss": 0.2813, "total_loss": 0.7462, "alive_pct": 13.3, "stability": 0.843},
        {"expansion": 2, "topk": 256, "recon_loss": 0.2286, "total_loss": 0.5255, "alive_pct": 46.9, "stability": 0.833},
        {"expansion": 4, "topk": 256, "recon_loss": 0.212, "total_loss": 0.6893, "alive_pct": 10.8, "stability": 0.836},
        {"expansion": 2, "topk": 64, "recon_loss": 0.2761, "total_loss": 0.5032, "alive_pct": 60.8, "stability": 0.831},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2825, "total_loss": 0.3311, "alive_pct": 96.5, "stability": 0.864},
        {"expansion": 1, "topk": 128, "recon_loss": 0.3341, "total_loss": 0.4653, "alive_pct": 80.0, "stability": 0.851},
        {"expansion": 1, "topk": 32, "recon_loss": 0.3456, "total_loss": 0.776, "alive_pct": 20.2, "stability": 0.869},
        {"expansion": 1, "topk": 64, "recon_loss": 0.3299, "total_loss": 0.4386, "alive_pct": 84.4, "stability": 0.862},
        {"expansion": 1, "topk": 256, "recon_loss": 0.3821, "total_loss": 0.5061, "alive_pct": 81.4, "stability": 0.863},
        {"expansion": 1, "topk": 64, "recon_loss": 0.3544, "total_loss": 0.7756, "alive_pct": 22.0, "stability": 0.856},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2721, "total_loss": 0.4288, "alive_pct": 74.9, "stability": 0.881},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2735, "total_loss": 0.4475, "alive_pct": 71.4, "stability": 0.885},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2781, "total_loss": 0.3317, "alive_pct": 95.5, "stability": 0.852},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2779, "total_loss": 0.3349, "alive_pct": 94.8, "stability": 0.849},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2762, "total_loss": 0.336, "alive_pct": 94.3, "stability": 0.848},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2732, "total_loss": 0.3538, "alive_pct": 90.1, "stability": 0.867},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2786, "total_loss": 0.3322, "alive_pct": 95.5, "stability": 0.841},
        {"expansion": 1, "topk": 256, "recon_loss": 0.2732, "total_loss": 0.3294, "alive_pct": 95.0, "stability": 0.829},
        {"expansion": 1, "topk": 32, "recon_loss": 0.3811, "total_loss": 0.6848, "alive_pct": 45.2, "stability": 0.834},
        {"expansion": 1, "topk": 128, "recon_loss": 0.3601, "total_loss": 0.5524, "alive_pct": 67.8, "stability": 0.849},
    ],
    "TabICL v2": [
        {"expansion": 8, "topk": 64, "recon_loss": 0.307, "total_loss": 0.5068, "alive_pct": 65.9, "stability": 0.915},
        {"expansion": 16, "topk": 16, "recon_loss": 0.3554, "total_loss": 0.7046, "alive_pct": 33.9, "stability": 0.848},
        {"expansion": 4, "topk": 16, "recon_loss": 0.4454, "total_loss": 0.565, "alive_pct": 82.3, "stability": 0.87},
        {"expansion": 4, "topk": 16, "recon_loss": 0.4536, "total_loss": 0.6108, "alive_pct": 74.3, "stability": 0.853},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0859, "total_loss": 0.1753, "alive_pct": 88.4, "stability": 0.795},
        {"expansion": 8, "topk": 16, "recon_loss": 0.3924, "total_loss": 0.5727, "alive_pct": 69.0, "stability": 0.915},
        {"expansion": 16, "topk": 32, "recon_loss": 0.3231, "total_loss": 0.701, "alive_pct": 30.5, "stability": 0.892},
        {"expansion": 16, "topk": 64, "recon_loss": 0.2402, "total_loss": 0.4305, "alive_pct": 67.3, "stability": 0.792},
        {"expansion": 8, "topk": 64, "recon_loss": 0.2364, "total_loss": 0.3454, "alive_pct": 84.3, "stability": 0.868},
        {"expansion": 16, "topk": 16, "recon_loss": 0.4297, "total_loss": 0.917, "alive_pct": 8.8, "stability": 0.906},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0649, "total_loss": 0.4375, "alive_pct": 31.7, "stability": 0.823},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1402, "total_loss": 0.2373, "alive_pct": 86.8, "stability": 0.868},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1496, "total_loss": 0.2427, "alive_pct": 87.6, "stability": 0.864},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1498, "total_loss": 0.2385, "alive_pct": 88.5, "stability": 0.86},
        {"expansion": 4, "topk": 256, "recon_loss": 0.1008, "total_loss": 0.4487, "alive_pct": 36.7, "stability": 0.777},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0674, "total_loss": 0.4865, "alive_pct": 22.4, "stability": 0.836},
        {"expansion": 8, "topk": 128, "recon_loss": 0.1335, "total_loss": 0.4686, "alive_pct": 39.1, "stability": 0.846},
        {"expansion": 8, "topk": 32, "recon_loss": 0.3088, "total_loss": 0.4466, "alive_pct": 78.2, "stability": 0.929},
        {"expansion": 16, "topk": 256, "recon_loss": 0.1607, "total_loss": 0.3453, "alive_pct": 69.3, "stability": 0.828},
        {"expansion": 4, "topk": 128, "recon_loss": 0.2005, "total_loss": 0.2588, "alive_pct": 94.6, "stability": 0.829},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0874, "total_loss": 0.2216, "alive_pct": 79.4, "stability": 0.855},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0945, "total_loss": 0.2097, "alive_pct": 83.2, "stability": 0.856},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0842, "total_loss": 0.2827, "alive_pct": 66.5, "stability": 0.86},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0769, "total_loss": 0.453, "alive_pct": 31.0, "stability": 0.863},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0829, "total_loss": 0.223, "alive_pct": 78.2, "stability": 0.853},
        {"expansion": 16, "topk": 256, "recon_loss": 0.084, "total_loss": 0.3999, "alive_pct": 43.1, "stability": 0.871},
        {"expansion": 16, "topk": 256, "recon_loss": 0.0955, "total_loss": 0.5107, "alive_pct": 23.2, "stability": 0.851},
        {"expansion": 16, "topk": 256, "recon_loss": 0.1304, "total_loss": 0.2649, "alive_pct": 79.3, "stability": 0.837},
        {"expansion": 16, "topk": 32, "recon_loss": 0.2917, "total_loss": 0.6702, "alive_pct": 28.7, "stability": 0.872},
        {"expansion": 16, "topk": 256, "recon_loss": 0.1334, "total_loss": 0.1856, "alive_pct": 95.8, "stability": 0.851},
    ],
}

embed_dims = {"TabICL": 512, "TabPFN": 192, "Mitra": 512, "CARTE": 300,
              "HyperFast": 776, "TabDPT": 726, "Tabula-8B": 4096, "TabICL v2": 512}

models = list(data.keys())
nmodels = len(models)
ncols = 3
nrows = (nmodels + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
axes = axes.flatten()
# Hide unused subplots
for i in range(nmodels, len(axes)):
    axes[i].set_visible(False)

colors = {"TabPFN": "#1f77b4", "TabICL": "#ff7f0e", "Mitra": "#2ca02c", "CARTE": "#d62728",
          "HyperFast": "#9467bd", "TabDPT": "#8c564b", "Tabula-8B": "#e377c2",
          "TabICL v2": "#bcbd22"}

for idx, model in enumerate(models):
    ax = axes[idx]
    trials = data[model]
    ed = embed_dims[model]

    complexities = [t["expansion"] * ed for t in trials]
    recon = [t["recon_loss"] for t in trials]
    total = [t["total_loss"] for t in trials]
    alive_pcts = [t["alive_pct"] for t in trials]

    # Color by alive_pct using a diverging colormap (red=dead, green=alive)
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.RdYlBu

    # Plot recon loss (filled) and total loss (hollow)
    sc = ax.scatter(complexities, recon, c=alive_pcts, cmap=cmap, norm=norm,
                    s=80, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3,
                    label="Recon loss")
    ax.scatter(complexities, total, c=alive_pcts, cmap=cmap, norm=norm,
               s=40, alpha=0.4, edgecolors="k", linewidths=0.3, marker="x",
               zorder=2, label="Total loss")

    ax.set_xlabel(f"Hidden dim (expansion × {ed})")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model} (embed={ed})", fontsize=14, fontweight="bold", color=colors[model])
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Auto-scale x-axis based on data
    xmax = max(complexities) * 1.1
    ax.set_xlim(-xmax * 0.03, xmax)
    # Nice tick labels
    tick_vals = sorted(set(complexities))
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{v//1000}K" if v >= 1000 else str(v) for v in tick_vals],
                       fontsize=7, rotation=45)

    # Add normalized 0 (best) to 1 (worst) scale on right y-axis
    rmin, rmax = min(recon), max(recon)
    ax2 = ax.twinx()
    ax2.set_ylim((ax.get_ylim()[0] - rmin) / (rmax - rmin),
                 (ax.get_ylim()[1] - rmin) / (rmax - rmin))
    ax2.set_ylabel("Normalized (0=best, 1=worst)", fontsize=9, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)

    # Annotations for three key configs:
    # 1. Best recon (gray) — lowest recon loss regardless
    # 2. Best alive recon (green) — lowest recon among alive>80%
    # 3. Best efficiency (blue) — minimizes recon × log2(complexity) among alive>80%
    annotated = set()

    # 1. Best recon
    best_idx = np.argmin(recon)
    t = trials[best_idx]
    ax.annotate(f"best recon\n{t['expansion']}x, k={t['topk']}, {t['alive_pct']:.0f}%",
                xy=(complexities[best_idx], recon[best_idx]),
                xytext=(15, 15), textcoords="offset points", fontsize=7,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    annotated.add(best_idx)

    # 2. Best alive recon (>80%)
    alive_mask = [a >= 85 for a in alive_pcts]
    if any(alive_mask):
        alive_recons = [r if m else 999 for r, m in zip(recon, alive_mask)]
        best_alive_idx = np.argmin(alive_recons)
        if best_alive_idx not in annotated:
            t2 = trials[best_alive_idx]
            ax.annotate(f"best alive\n{t2['expansion']}x, k={t2['topk']}, {t2['alive_pct']:.0f}%",
                        xy=(complexities[best_alive_idx], recon[best_alive_idx]),
                        xytext=(-60, 25), textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="green", alpha=0.8))
            annotated.add(best_alive_idx)

        # 3. Best efficiency: min recon × sqrt(hidden_dim), alive>85%
        efficiency = [r * np.sqrt(c) if m else 999
                      for r, c, m in zip(recon, complexities, alive_mask)]
        best_eff_idx = np.argmin(efficiency)
        if best_eff_idx not in annotated:
            t3 = trials[best_eff_idx]
            ax.annotate(f"best efficiency\n{t3['expansion']}x, k={t3['topk']}, {t3['alive_pct']:.0f}%",
                        xy=(complexities[best_eff_idx], recon[best_eff_idx]),
                        xytext=(-20, -30), textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="->", color="blue", lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="blue", alpha=0.8))

# Add shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label("Alive %", fontsize=11)

fig.suptitle("Round 8 SAE Sweeps: Hidden Dim vs Loss\n(color = alive %, circles = recon, × = total)",
             fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 0.92, 0.94])
plt.savefig("output/paper_figures/complexity_vs_loss_r8.pdf", bbox_inches="tight", dpi=150)
plt.savefig("output/paper_figures/complexity_vs_loss_r8.png", bbox_inches="tight", dpi=150)
print("Saved to output/paper_figures/complexity_vs_loss_r8.{pdf,png}")
