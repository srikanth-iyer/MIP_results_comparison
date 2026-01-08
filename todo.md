# TODOs for genx-notebooks

## DONES

DONE: Regional generation is not plotting correctly. CHECK THIS ASAP

DONE: check emissions plot check the zone numbering legend and check overall value and units

DONE: Capacity factor is probably a percentage. check this and add the % sign if so. The capacity)factor was being calculated somehow rather than using the exact value form the capacityfactor.csv file. fixed this to use the value verbatim.

DONE: check generators and mapping. print a csv file for generator names and mapping so i can look at it with npatankar

DONE: in dispatch, add the other categories in addition to wind. get zone labelling

DONE: generate an annual_demand.sv file with the right zone values and create a common file including all planning years for the generation_capacity plots where the black line is for the annual demand of that year.

DONE: incorporate new inputs data. check if year wise data can be plotted properly for the same scenario

DONE: Remove flex demand from capacity plots

DONE: remove the "Model" term from x axis

DONE:  remove the black lines from the bar plots

DONE: CLEAN UP THE REPO TO MAKE IT PEOPLE FACING

DONE: Add retrofit and hydrogen to the capacity factors and dispatch plots

DONE: Add transmission data table

DONE: take a sample netrevenue file and plot them

## CURRENT PRIORITY

TODO: operational nse like the one on greg's plots online (compare current policies file )

TODO: revenue & costs by technology sub tab add a marker or dot for total profit (merge the profit by tech tab onto this by adding a dot in the exiwting plot))

TODO: cost assumptions(input costs)(this is in the resources subfolder)), demand profiles, fuel prices

TODO: get map plotting from kavi

TODO: add the substation as a dot on the starting point and ending point of the transmission mapping lines 

## LOW PRIORITY

TODO: for genx visualizations, have an option to only specify a few regions in the regional plots
DONE: turn to a github page so everyone can view
DONE:  do transmission mapping
DONE: IMPORT/EXPORTS tab. imports into any NY region and export out of any NY region - x axis should be region. +ve y axis should show import and -ve y axis should show exports
