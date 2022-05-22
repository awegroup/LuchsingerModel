# Luchsinger's performance model of a pumping kite system accounting for tether elevation angle

## Model purpose
Preliminary sizing of a pumping kite system on Mars.

## Build Status
* Model is complete, but the sequence of execution is not optimal making the model slow.
* Having 669 sols (days on Mars) is hard coded in the model, requiring manual changes to simulate 365 days on Earth.
* The parameters and methods description is not up to date.

## Features
* Calculate the power of a kite of a certain size for given environment conditions [1].
* Models the mechanical to electrical energy conversion losses [2].
* Evaluates system mass and specifications.

## Framework used
The architecture of the model is outlined in the [code block diagram](doc/LuchsingerModel_CodeBlcokDriagram.pdf).


## Installation
Required Python packages might need to be installed.

## How to Use
1. Change input as desirable.
2. Run YearlyPower_elevation.py for a yearly simulation (Option 1) or PowerPerSol_elevation.py for simulating only sols of interest (Option 2).

* Option 1 produces a power curve for the year for a kite (combined results shown [here](doc/Wind_power_for_kites.png)).
* Option 2 produces operational evenvelope of pumping kite for sols (combined results shown [here](doc/Operational_envelope_kite.png)).

## Possible contributions   
* Instead of evaluating the complete state of each sol one after another, it will be faster if some steps are done for a year.
* Examples: instead of updating the density of each sol though the python classes, a value from a list with 669 points should be used.
* Instead of updating the k and u Weibull parameters of each sol though the python classes, a value from a list with 669 points should be used.

## Credits
TU Delft DSE group 23 of 2020 created the first version of this model.

## Authors
Lora Ouroumova

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details

## References

[1] R.H. Luchsinger: “Pumping cycle kite power”. In U. Ahrens, M. Diehl, and R. Schmehl (eds), Airborne Wind Energy, Green Energy and Technology, chapter 3, pages 47–64. Springer, Berlin Heidelberg, 2013. https://doi.org/10.1007/9783642399657_3

[2] U. Fechner, R. Schmehl: “Model-Based Efficiency Analysis of Wind Power Conversion by a Pumping Kite Power System”. In U. Ahrens, M. Diehl, and R. Schmehl (eds), Airborne Wind Energy, Green Energy and Technology, chapter 14, pages 249–270. Springer, Berlin Heidelberg, 2013. https://doi.org/10.1007/9783642399657_14

[3] L. Ouroumova et al.:
Ouroumova, L., Witte, D., Klootwijk, B., Terwindt, E., van Marion, F., Mordasov, D., Corte Vargas, F., Heidweiller, S., Géczi, M., Kempers, M., & Schmehl, R.  “Combined Airborne Wind and Photovoltaic Energy System for Martian Habitats”. In: Spool 8.2 (2021), pp. 71–85. http://doi.org/10.7480/spool.2021.2.6058

[4] Corte Vargas, F., Géczi, M., Heidweiller, S., Kempers, M.X., Klootwijk, B.J., van Marion, F., Mordasov, D., Ouroumova, L.H., Terwindt, E.N., Witte, D. (2020) Arcadian Renewable Energy System: Renewable Energy for Mars Habitat [Technical report]. Faculty of Aerospace Engineering, Delft University of Technology. http://resolver.tudelft.nl/uuid:93c343e5-ee79-4320-98a3-949d3e9c407d
