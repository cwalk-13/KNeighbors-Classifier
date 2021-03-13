import mysklearn.myutils as myutils
# TODO: copy your mypytable.py solution from PA2 here
"""
Charles Walker 
CPSC 322
Section 02
PA2 
"""

import copy
import csv 
import statistics
from statistics import mode
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        M = 0
        N = 0
        for i in range(len(self.data[0])): 
            M += 1
        
        for i in range(len(self.data)):
            N += 1
        return N, M # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            tuple of int: rows, cols in the table
        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = 0
        if isinstance(col_identifier, str):
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int):
            col_index = col_identifier
        else:
            print("invalid Column identifier")
            pass
        
        col = []

        if (include_missing_values == False):
            for row in self.data:
                if (row[col_index] != "NA" and row[col_index] != ""):
                    col.append(row[col_index])

        if (include_missing_values == True):
            for row in self.data:
                col.append(row[col_index])

        return col # TODO: fix this

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        row_length = len(self.column_names)
        if row_length == 1:
            for i in range(len(self.data)):
                try: 
                    numeric_value = float(self.data[i])
                    self.data[i] = numeric_value
                except ValueError:
                    pass
        if row_length > 1:
            for i in range(len(self.data)):
                for j in range(row_length):
                    try: 
                        numeric_value = float(self.data[i][j])
                        self.data[i][j] = numeric_value
                    except ValueError:
                        pass
        pass # TODO: fix this

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for i in range(len(rows_to_drop)):
            self.data.remove(rows_to_drop[i])

        pass # TODO: fix this

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # TODO: finish this
        infile = open(filename, encoding="utf8")
        lines = csv.reader(infile)
        for line in lines:
            self.data.append(line)
        infile.close()
        self.column_names = self.data.pop(0)
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        for i in range(len(self.column_names) - 1):
            outfile.write(str(self.column_names[i]) + ",")
        outfile.write(str(self.column_names[i + 1]) + "\n")

        for row in self.data:
            for i in range(len(row) - 1):
                outfile.write(str(row[i]) + ",")
            outfile.write(str(row[i + 1]) + "\n")

        outfile.close()
        pass # TODO: fix this

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.

        find first row with key
        find subsequent rows with the same key and append
        
        """
        duplicates = []
        cols_i = []
        for key in key_column_names:
            col_index = self.column_names.index(key)
            cols_i.append(col_index)
        col_keys = []
        for i in range(len(self.data)):
            skip = False
            first_inst = self.data[i]
            first_inst_keys = []
            for col in cols_i:
                first_inst_keys.append(first_inst[col])
            for key in col_keys:
                if first_inst_keys == key:
                    skip = True
            if skip == True:
                continue
            col_keys.append(first_inst_keys)
            for j in range(i +1, len(self.data)):
                next_inst_keys = []
                next_inst = self.data[j]
                for col in cols_i:
                    next_inst_keys.append(next_inst[col])
                if next_inst_keys == first_inst_keys:
                    duplicates.append(self.data[j])

        return duplicates # TODO: fix this

    def ordered_col(self, key_column_names, include_first=True):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.

        find first row with key
        find subsequent rows with the same key and append

        """
        duplicates = []
        col_keys = []
        for i in range(len(self.data)):
            skip = False
            first_inst = str(self.data[i])
            for key in col_keys:
                if first_inst == key:
                    skip = True
            if skip == True:
                continue
            col_keys.append(first_inst)
            #includes first instance
            for j in range(i, len(self.data)):
                next_inst = str(self.data[j])
                if next_inst == first_inst:
                    duplicates.append(str(self.data[j]))

        return duplicates # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        row_length = len(self.column_names)
        table_copy = []
        
        for i in range(len(self.data)):
            has_NA = False
            for j in range(row_length):
                if self.data[i][j] == "NA" or self.data[i][j] == "":
                    has_NA = True
            if has_NA == False:  
                table_copy.append(self.data[i])
        
        self.data.clear()
        self.data = copy.deepcopy(table_copy)   
        pass # TODO: fix this

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.get_column(col_name)

        header = [col_name]
        col_mypy = MyPyTable(header, col)
        col_mypy.convert_to_numeric()
        
        col_avg = 0
        for i in range(len(col_mypy.data)):
            if col_mypy.data[i] != "NA" and col_mypy.data[i] != "":
                col_avg += col_mypy.data[i]
                print(col_avg)
        col_avg = col_avg/(i)
        print(col_avg)
        col_id = self.column_names.index(col_name)
        for i in range(len(self.data)):
            if self.data[i][col_id] == "NA" or self.data[i][col_id] == "":
                self.data[i][col_id] = col_avg 
        

        pass # TODO: fix this

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
        self.convert_to_numeric()
        stats = []
        for col_name in col_names:
            stat_row = []
            col = self.get_column(col_name, False)
            stat_row.append(col_name)
            if len(col) > 0:                
                col_min = min(col)
                col_max = max(col)
                col_mid = col_min + (col_max - col_min)/2
                stat_row.append(col_min)
                stat_row.append(col_max)
                stat_row.append(col_mid)
                col_avg = 0
                for i in range(len(col)):
                    col_avg += col[i]
                col_avg = col_avg/(i+1)
                stat_row.append(col_avg) 
                col.sort()
                if len(col)%2 == 0:
                    col_med1 = col[len(col)//2]
                    print (col_med1)
                    col_med2 = col[len(col)//2 - 1]
                    print(col_med2)
                    col_med = (col_med1 + col_med2) / 2
        
                if len(col)%2 != 0:
                    col_med = col[len(col)//2]            
                stat_row.append(col_med)
                stats.append(stat_row)
            else:
                for i in range(5):
                    stat_row.append("NA")


        header = ["attribute", "min", "max", "mid", "avg", "median"]

        return MyPyTable(header, stats) # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.

        . find the rows with matching keys, join them
        for row in table1
            *skip if there is no match found
            for row in table2
                if match    
                    join the rows and add to table3
        """
        
        cols_i = []
        for key in key_column_names:
            col_index = self.column_names.index(key)
            cols_i.append(col_index)
        
        other_cols_i = []
        for key in key_column_names:
            col_index = other_table.column_names.index(key)
            other_cols_i.append(col_index)
        
        table3 = []

        for i in range(len(self.data)):
            table1_row = self.data[i]
            table1_row_keys = []
            for col in cols_i:
                table1_row_keys.append(table1_row[col])
            for j in range(len(other_table.data)):
                table2_row = other_table.data[j]
                table2_row_keys = []
                for col in other_cols_i:
                    table2_row_keys.append(table2_row[col])
                if table2_row_keys == table1_row_keys:
                    joined_row = []
                    for col in self.data[i]:
                        joined_row.append(col)
                    for col in range(len(other_table.data[j])):
                        found = False
                        for col_key in other_cols_i:
                            if col == col_key:
                                found = True 
                        if found == False:        
                            joined_row.append(other_table.data[j][col])
                    table3.append(joined_row) 
        header = []
        for col in self.column_names:
            header.append(col)
        for col in range(len(other_table.column_names)):
            found = False
            for col_key in other_cols_i:
                if col == col_key:
                    found = True 
            if found == False:        
                header.append(other_table.column_names[col])
            


        return MyPyTable(header, table3) # TODO: fix this

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".

        . creates table with matches and non matches 
        make copy of table1 call it table3
        for row in table3
            for row in table2
                if match in table2
                    join table2's row information
                else 
                    fill missing vals with NA
        """
        
        cols_i = []
        for key in key_column_names:
            col_index = self.column_names.index(key)
            cols_i.append(col_index)
        
        other_cols_i = []
        for key in key_column_names:
            col_index = other_table.column_names.index(key)
            other_cols_i.append(col_index)
        print(other_cols_i)
        header = []
        for col in self.column_names:
            header.append(col)
        for col in range(len(other_table.column_names)):
            found = False
            for col_key in other_cols_i:
                if col == col_key:
                    found = True 
            if found == False:        
                header.append(other_table.column_names[col])

        table3 = copy.deepcopy(self.data)

        for i in range(len(table3)):
            table3_row = table3[i]
            table3_row_keys = []
            match = False
            for col in cols_i:
                table3_row_keys.append(table3_row[col])
            for j in range(len(other_table.data)):
                table2_row = other_table.data[j]
                table2_row_keys = []
                for col in other_cols_i:
                    table2_row_keys.append(table2_row[col])
                if table2_row_keys == table3_row_keys:
                    for col in range(len(other_table.data[j])):
                        found = False
                        for col_key in other_cols_i:
                            if col == col_key:
                                found = True 
                        if found == False:        
                            table3[i].append(other_table.data[j][col])
                            match = True
            if match == False:
                cols = len(other_table.column_names) - len(key_column_names)
                for col in range(cols):
                    table3[i].append("NA")

    
        for i in range(len(other_table.data)):
            table2_row = other_table.data[i]
            table2_row_keys = []
            match = False
            for col in other_cols_i:
                table2_row_keys.append(table2_row[col])
            for j in range(len(table3)):
                table3_row = table3[j]
                table3_row_keys = []
                for col in cols_i:
                    table3_row_keys.append(table3_row[col])
                if table2_row_keys == table3_row_keys:
                    match = True
            if match == False:
                new_row = []
                for col in range(len(header)):
                    col_val = "NA"
                    for it in range(len(other_table.column_names)):
                        if other_table.column_names[it] == header[col]:
                            col_val = other_table.data[i][it]

                    new_row.append(col_val)
                table3.append(new_row)

        return MyPyTable(header, table3) # TODO: fix this