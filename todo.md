DONE: Regional generation is not plotting correctly. CHECK THIS ASAP

DONE: check emissions plot check the zone numbering legend and check overall value and units

DONE: Capacity factor is probably a percentage. check this and add the % sign if so. The capacity)factor was being calculated somehow rather than using the exact value form the capacityfactor.csv file. fixed this to use the value verbatim.

DONE: in dispatch, add the other categories in addition to wind. get zone labelling

DONE: generate an annual_demand.sv file with the right zone values and create a common file including all planning years for the generation_capacity plots where the black line is for the annual demand of that year.

DONE: incorporate new inputs data. check if year wise data can be plotted properly for the same scenario

TODO: Add transmission data table

TODO: operational nse like the one on greg's plots online (compare current policies file )

TODO: Seperate tab for cost related resuls and add costs by zone (operational costs)
TODO: check all operational results plots some look iffy
TODO: OPERATIONAL costs check the diff between model and society

LOW PRIORITY
TODO: for genx visualizations, have an option to only specify a few regions in the regional plots
TODO: turn to a github page so everyone can view

TODO:  do transmission mapping

TODO: IMPORT/EXPORTS tab. imports into any NY region and export out of any NY region - x axis should be region. +ve y axis should show import and -ve y axis should show exports
