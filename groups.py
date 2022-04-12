control = ['15', '20', '21', '22', '24', '25', '29', '32', '33', '44', '46', '50', '53', '55', '56', '57', '60', '69',
           '71', '73', '74', '77', '78', '79', '81', '87', '88', '91', '94', '101', '105']
mania = ['04', '06', '08', '10', '11', '13', '17', '23', '34', '35', '36', '37', '40', '42', '61', '67', '83', '97',
         '111', '112', '113', '116', '119', '128', '129', '133', '135', '136', '137']
mixed_mania = ['07', '12', '14', '19', '47', '62', '65', '80', '82', '96', '107', '130']
mixed_depression = ['09', '16', '18', '28', '45', '49', '66', '84', '85', '89', '95', '104', '106', '114', '121', '123',
                    '125', '131', '134', '138', '140', '142']
depression = ['01', '02', '26', '27', '38', '39', '41', '54', '59', '64', '72', '75', '76', '90', '92', '98', '100',
              '102', '109', '110', '120', '124', '126', '127', '132', '141']
euthymia = ['03', '05', '30', '31', '43', '48', '51', '52', '58', '63', '68', '70', '86', '93', '99', '103', '115',
            '118', '122', '139']

groups = [control, mania, mixed_mania, mixed_depression, depression, euthymia]
groupnames = ['control', 'mania', 'mixed mania', 'mixed depression', 'depression', 'euthymia']
groups_and_names = zip(groups, groupnames)
names_to_groups_dict = {name: group for (group, name) in groups_and_names}

all_numbers = control + mania + mixed_mania + mixed_depression + depression + euthymia


def number_to_group(number: str) -> str:
    for group in groups:
        if number in group:
            for name in groupnames:
                if names_to_groups_dict[name] == group:
                    return name

