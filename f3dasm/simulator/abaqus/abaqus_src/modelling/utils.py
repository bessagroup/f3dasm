'''
Created on 2020-04-15 18:01:32
Last modified on 2020-09-22 14:59:16
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Place where additional useful functions are defined.
'''


#%% write to inp

class AddToInp(object):

    def __init__(self, text, job_name, section='OUTPUT REQUESTS'):
        '''
        Parameters
        ----------
        text : array of str
            Each component is a line to be added.
        section : str
            Section where new text must be placed.
        '''
        # store variables
        self.text = text
        self.job_name = job_name
        self.section = section
        # create useful variables
        self.filename = ('%s.inp' % job_name)
        self.text.append('**')

    def write_text(self):

        # get insert line
        lines, insert_line = self._find_line()

        # insert text
        for line in reversed(self.text):
            lines.insert(insert_line, '{}\n'.format(line))

        # write text
        with open(self.filename, 'w') as f:
            f.writelines(lines)

    def _find_line(self):

        # read file
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        # find section line
        line_cmp = '** {}\n'.format(self.section.upper())
        for i, line in reversed(list(enumerate(lines))):
            if line == line_cmp:
                break

        return lines, i + 2
