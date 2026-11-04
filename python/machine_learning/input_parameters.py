"""Input-parameter (feature) lists for the three descriptors used in the paper.

The column order matters for the multilayer perceptron (it fixes which initial
weight connects to which feature), so do not reorder these lists.

    Atomic Descriptor             -> atom_distance_charge_spin_input_parameters (5)
    Wyckoff Letter Descriptor     -> sic_wyckoff_letter_input_parameters        (15)
    Wyckoff Coordinate Descriptor -> sic_input_parameters                       (59)

The "Compare" descriptors (*_nw_data) use the Wyckoff lists on the samples of
the Atomic dataset.
"""

atom_distance_charge_spin_input_parameters = ['vac_sub_defect', 'vac_sub_defect_2', 'distance', 'charge', 'spin']

sic_wyckoff_letter_input_parameters = [
    'vac_sub_defect_h',
    'vac_sub_defect_h_2',
    'vac_sub_defect_k',
    'vac_sub_defect_k_2',
    "a",
    "a_2",
    "b",
    "b_2",
    "c",
    "c_2",
    "d",
    "d_2",
    'distance',
    'charge',
    'spin',
]


sic_input_parameters = [
    'vac_sub_defect_h',
    'vac_sub_defect_h_2',
    'vac_sub_defect_k',
    'vac_sub_defect_k_2',
    '(0, 0, 3/32)',
    '(0, 0, 3/32)_2',
    '(0, 0, 11/32)',
    '(0, 0, 11/32)_2',
    '(1/3, -1/3, 3/32)',
    '(1/3, -1/3, 3/32)_2',
    '(1/3, -1/3, 11/32)',
    '(1/3, -1/3, 11/32)_2',
    '(1/3, -1/3, 19/32)',
    '(1/3, -1/3, 19/32)_2',
    '(1/3, -1/3, 27/32)',
    '(1/3, -1/3, 27/32)_2',
    '(19/99, -19/99, 3/32)',
    '(19/99, -19/99, 3/32)_2',
    '(45/82, -45/82, 8/85)',
    '(45/82, -45/82, 8/85)_2',
    '(37/82, -37/82, 8/85)',
    '(37/82, -37/82, 8/85)_2',
    '(2/7, 2/7, 11/32)',
    '(2/7, 2/7, 11/32)_2',
    '(1/3, -1/3, 7/32)',
    '(1/3, -1/3, 7/32)_2',
    '(1/3, -1/3, 31/32)',
    '(1/3, -1/3, 31/32)_2',
    '(5/6, -5/6, 3/32)',
    '(5/6, -5/6, 3/32)_2',
    '(21/88, -21/88, 7/32)',
    '(21/88, -21/88, 7/32)_2',
    '(7/74, -7/74, 7/32)',
    '(7/74, -7/74, 7/32)_2',
    '(4/7, -4/7, 7/32)',
    '(4/7, -4/7, 7/32)_2',
    '(3/7, -3/7, 7/32)',
    '(3/7, -3/7, 7/32)_2',
    '(55/96, -55/96, 15/32)',
    '(55/96, -55/96, 15/32)_2',
    '(41/96, -41/96, 15/32)',
    '(41/96, -41/96, 15/32)_2',
    '(51/67, -51/67, 15/32)',
    '(51/67, -51/67, 15/32)_2',
    '(48/53, -48/53, 15/32)',
    '(48/53, -48/53, 15/32)_2',
    '(1/3, -1/3, 13/32)',
    '(1/3, -1/3, 13/32)_2',
    '(1/3, -1/3, 29/32)',
    '(1/3, -1/3, 29/32)_2',
    '(5/6, -5/6, 7/32)',
    '(5/6, -5/6, 7/32)_2',
    '(1/2, -1/2, 13/32)',
    '(1/2, -1/2, 13/32)_2',
    '(1/6, -1/6, 15/32)',
    '(1/6, -1/6, 15/32)_2',
    'distance',
    'charge',
    'spin',
]


FEATURES = {
    "sic_no_wyckoff": atom_distance_charge_spin_input_parameters,
    "sic_wyckoff_letter": sic_wyckoff_letter_input_parameters,
    "sic_wyckoff": sic_input_parameters,
    "sic_wyckoff_letter_nw_data": sic_wyckoff_letter_input_parameters,
    "sic_wyckoff_nw_data": sic_input_parameters,
}
