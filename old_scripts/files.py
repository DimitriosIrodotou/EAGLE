# Import required python libraries #
import re

# File parameters of the G-EAGLE data #
first_file = 0
last_file = 255
file_prefix = 'eagle_subfind_tab_'
output_path = '/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_data/0021/groups_010_z005p000/'
run, file_number, redshift = [item.replace('/', '') for item in re.split('_data/|/groups_|_', output_path)[1:4]]
file = output_path + file_prefix + file_number + '_' + redshift + '.'
title = run + redshift