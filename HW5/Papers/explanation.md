## <center> CSE 256: Data Mining and Analytics </center> <center> Assignment 5: Wealth Prediction </center>
### Leaderboard Name: `Potato`

### Members:  
1. Hrishikesh Bharadwaj Chakrapani - A59002326
2. Prajwal Yelandur-Raghu - A59008845
3. Pratyush Karmakar - A59012917
4. Raghasuma Aishwarya Putchakayala - A59002409

### Approach: Feature Engineering
Our approach mainly consists of excluding KDTree encoding and improving our model's performance using feature engineering. We trained the model on the following features: 
 1. Bare soil indices:
 - Modified Bare Soil Index, MBI = $\frac{\text{SWIR1 - SWIR2 - NIR}}{\text{SWIR1 + SWIR2 + NIR}} + 0.5$
 - Bare Soil Index, BSI = $\frac{(\text{SWIR2 + R}) - (\text{NIR + B})}{(\text{SWIR2 + R}) + (\text{NIR + B})}$
 - Bare Soil Index 1, BSI1 = $\frac{(\text{SWIR1 + R}) - (\text{NIR + B})}{(\text {SWIR1 + R)} + (\text{NIR + B})}$
 - Bare Soil Index 2, BSI2 = $100 \times \sqrt{\frac{\text{SWIR2 - G}}{\text{SWIR2 + G}}}$
 - Bare Soil Index 3, BSI3 = $\frac{(\text{SWIR1 + R}) - (\text{NIR + B})}{(\text{SWIR1 + R)} + (\text{NIR + B})} \times 100 + 100$
 - Normalized difference soil index 1, NDSI1 = $\frac{\text{SWIR1 - NIR}}{\text{SWIR1 + NIR }}$
 - Normalized difference soil index 2, NDSI 2 = $\frac{\text{SWIR2 - G}}{\text{SWIR2 + G}}$
 - Bareness Index, BI = $(\text{R + SWIR1 - NIR})$
 - Dry bare-soil index, DBSI=$\frac{\text{SWIR1 - G}}{\text{SWIR1 + G}} - \frac{\text{NIR - R}}{\text{NIR + R}}$
 2. Statistical features:
 - Mean of values from each channel
 - Standard deviation of values from each channel
 - Median of values from each channel
 - Count of pixels that are 1.5 standard deviations above the mean indicating the brightness of the area (outliers)
 - Count of pixels that are 1.5 standard deviations below the mean indicating the darkness of the area (outliers)
 3. Urban or Rural
 4. Multispectral indices:
 - The Built-up Area Extraction Index, BAEI = $\frac{(\text{RED + 0.3})} {(\text{GREEN + SWIR1})}$
 - The Built-up Index, BUI = $\frac{(\text{SWIR1 - NIR})}{(\text{SWIR1 + NIR})} - \frac{(\text{NIR - RED})}{(\text{NIR + RED})}$
 -  The New Built-up Index, NBI = $\frac{(\text{RED - SWIR1})}{(\text{NIR})}$
 -  The Band Ratio for Built-up Area, BRBA = $\frac{(\text{RED})}{(\text{SWIR1})}$ 
 - The Index-based Built-up Index, IBI = $\frac{\text{2 * SWIR1}}{(\text{SWIR1 + NIR})} - \frac{[\frac{\text{NIR}}{(\text{NIR - RED})} + \frac{\text{GREEN}}{(\text{GREEN + SWIR1})}]}{2} *\frac{\text{SWIR1}}{(\text{SWIR1 + NIR})} + [\frac{\text{NIR}}{(\text{NIR - RED})} + \frac{\text{GREEN}}{(\text{GREEN + SWIR1})}]$
 - The Modified Built-up Area Index, MBAI = $\frac{[\text{NIR + (1.57 * GREEN) + (2.4 * SWIR1)}]}{(\text{1 + NIR})}$
 - The Normalized Difference Concrete Condition Index, NDCCI =  $\frac{(\text{NIR - GREEN})}{(\text{NIR + GREEN})}$
5. Dominant colours and counts - Colours are clustered based on the RGB values and the top 5 dominant colours' RGB values are considered for features (15 features). The frequency of the pixels that contain these top 5 dominant colours are used as features (5 features) - total of 20 features

### References

 1. A Modified Bare Soil Index to Identify Bare Land Features during Agricultural Fallow-Period in Southeast Asia Using Landsat 8: https://www.mdpi.com/2073-445X/10/3/231
 2. Classification of Urban Area Using Multispectral Indices for Urban Planning: https://www.mdpi.com/2072-4292/12/15/2503 