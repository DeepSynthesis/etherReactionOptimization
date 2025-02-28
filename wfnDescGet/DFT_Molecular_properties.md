# [List of Usable DFT molecular descriptors](http://sobereva.com/601)

## [ESP (electrostatic potential) properties](http://bbs.keinsci.com/thread-219-1-1.html)

### [GIPF (general interaction properties function)](https://doi.org/10.1016/0166-1280(94)80117-7)

1. maxium and minium of surface ESP
2. average/variance of positive/negative/all surface ESP
3. degree of charge balance
4. [molecular polarity index](http://sobereva.com/518)

### surface properties

1. area of positive/negative surface

### specific atom ESP

1. average ESP of specific atom's van der Waals surface 

### other things of ESP

1. average local ionization energy(ALIE)
2. local electron attach energy(LEAE)
3. local electron affinity(LEA)

## [Concept DFT(CDFT) & reactivity](http://bbs.keinsci.com/thread-384-1-1.html)
> also can see in Multiwfn manual 3.25
### [Global molecular properties](http://sobereva.com/484)

1. Vertical ionization potential (VIP)
2. Vertical electron affinity (VEA)
3. Mulliken electronegativity
4. Chemical potential
5. Electron hardness
6. Electron softness
7. Electrophilicity index
8. Electrophilicity index-2(ω<sub>cubic</sub>)
9. Nucleophilicity index

### [Real space function](http://sobereva.com/484)

1. Fukui function (cannot be applied in descriptors)
2. Dual descriptor (cannot be applied in descriptors)
3. Local softness
4. Local electrophilicity index
5. Local nucleophilicity index

### [Atomic properties]((http://sobereva.com/484))

1. Condensed Fukui function
2. Condensed dual descriptor
3. Condensed local softness
4. Relative electrophilicity index
5. Relative nucleophilicity index
6. Condensed local electrophilicity index
7. Condensed local nucleophilicity index
8. Condensed local ω<sub>cubic</sub> electrophilicity index

## Molecular Structure

1. Molecular bond length, angle and dihedral angle properties
2. [Molecular radius](http://sobereva.com/190): defined as the closest/farthest distance from the geometric center to the molecular surface
2. [Molecular volumn and surface area](http://sobereva.com/102): calculate by Marching Tedrahedron algorithm
4. [Molecular length, width and height](http://sobereva.com/426): 

## Atom Descriptos

### Atom charge

1. Hirshfeld
2. Mulliken
3. SCPA
4. Modified Mulliken by Bickelhaupt
5. Becke 
6. ADCH
7. CHELPG-ESP
8. MK-ESP
9. AIM: cannot calculate hear....
10. CM5
11. RESP

### Atom oxidation state

1. [LOBA](http://sobereva.com/362)

### Atom basin analysis (not use)

> If Multiwfn in parallel mode, the index of basin does not correspond to atom number.
> Need to align it with atom cordination which is complex...

1. atom charge
2. ESP, ELF, electron density etc.

### Atom real space function 1-ai-···-q

1. Total ESP (related to pka)
> there are some other functions but still not know what's the use

### Othres

1. atom dipole
2. energy index

## [Bond properties](http://sobereva.com/471)

### AIM analysis
> same problem... need to align CP with bond
1. rho(BCP) & V(BCP)
2. ...

### Bond order
1. Mayer bond oder 1
2. Wiberg bond order 3
3. Mulliken bond order 4
4. Fuzzy bond order analysis (FBO) 7
4. Laplacian bond order (LBO) 8

### Bond dipole
200-2-...




## Other descriptors

1. HOMO/LUMO/HOMO-LUMO gap: 0
2. [HOMO/LUMO component (SCPA method)](http://sobereva.com/131): 8-3-h-l-0--10
3. [Orbital delocalization index, ODI](http://sobereva.com/525)
3. [Electronic spatial extent](http://sobereva.com/616): 
4. dipole of hole molecular
5. [(hyper)polarizability](http://sobereva.com/231): need `polar` keywords in gaussian...
