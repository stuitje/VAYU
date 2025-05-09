Basic atmosphere composition .toml files for generated config files. Includes:

p_surf: surface pressure of atmosphere.
p_top: top pressure of atmosphere.
vmr_dict: composition of atmosphere.
transparent: for bare-rock modelling, the atmosphere can be set to transparent. Replaces are beforementioned variables. 

Example structure:

    title = "1 bar CO2"

    [composition]
    p_surf = 1
    p_top = 1e-3
    vmr_dict = {CO2 = 1.0}

Bare-rock:

    title = "bare_rock"

    [composition]
    transparent	= true

