import numpy
import operator
from ..extern import six, pyparsing as pp

_builtins = {}
_builtins['nan'] = numpy.nan
_builtins['inf'] = numpy.inf
_builtins['log'] = numpy.log
_builtins['log10'] = numpy.log10
_builtins['exp'] = numpy.exp

class SelectionError(Exception):
    pass
    
# helper classes that are compiled at parse time
# see http://pyparsing.wikispaces.com/file/view/simpleBool.py
class CompOperand(object):
    """
    Base class for comparison operands
    """
    def __init__(self, r):
        """
        Parameters
        ----------
        r : ParseResults
            the ParseResults instance; must have unity length
        """
        self.value = r[0]
    
    def eval(self, data):
        """
        The main function that does the work for each operand, given
        input data array
        
        Parameters
        ----------
        data : array_like
            array that has named fields, i.e., a structured array
            or DataFrame
        """
        raise NotImplementedError
    
    def __str__(self):
         return str(self.value)
    __repr__ = __str__
    
class LeftCompOperand(CompOperand):
    """
    The left operand that gives the name of the column. This 
    class supports vector indexing syntax, similar to numpy.
    
    This class is responsible for returning the specific column 
    (possibly sliced) from the input data array    
    
    Examples
    --------
    >> "LogMass < 12"
    >> "Position[:,0] < 1000.0"
    >> log10(Mass) < 12
    """
    def __init__(self, r):
        super(LeftCompOperand, self).__init__(r)
        self.function = None
        
        # syntax of passed string was "function(column)" -- extract function
        if len(self.value) > 2: 
            if self.value[0] not in _builtins:
                raise SelectionError("function '%s' operating on selected column not recognized" %self.value[0])
            self.function = _builtins[self.value[0]]
            self.value = self.value[2]
            
        self.column_name = self.value[0]
        self.index = None if self.value[1].isempty() else self.value[1]
            
    def eval(self, data):
        """
        Return the named column from the input data
        
        Parameters
        ----------
        data : array_like
            array that has named fields, i.e., a structured array
            or DataFrame
        
        Returns
        -------
        array_like : 
            the specific column of the input data
        """
        
        # first try to get the data to return
        try:
            toret = data[self.column_name]
        except Exception as e:
            args = (self.column_name, str(e))
            raise SelectionError("Left access : cannot access column '%s' in input data array (%s)" %args)
         
        # now try to slice
        if self.index is not None:
            # data should be multidimensional
            if numpy.ndim(toret) == 1:
                raise SelectionError("Left access : array indexing should be used on multi-dimensional data columns")
                
            # try to eval the index string
            try:
                toret = eval("toret%s" %self.index, {'toret': toret})
            except Exception as e:
                args = (self.index, str(e))
                raise SelectionError("Left access cannot slice data using index '%s' (%s)" %args)

        # check the dimension -- must be unity
        if numpy.ndim(toret) != 1:
            raise SelectionError("Left access : result is not 1-dimensional")
        if hasattr(data, 'shape') and \
            data.shape[0] != toret.size:
            raise SelectionError("Left access : length mismatch between selection index and input data; maybe an indexing error?")

        # call the function, if it was passed
        if self.function is not None:
            toret = self.function(toret)
            
        return toret

class ColumnSlice(object):
    """
    The optional column index for the left-hand side of the query
    selection -- this will be applied by LeftCompOperand, if present
    """
    def __init__(self, r):
        self.value = "".join(r)
        
    def isempty(self):
        return self.value == ""
        
    def __str__(self):
        return str(self.value)
    __repr__ = __str__
    
class RightCompOperand(CompOperand):
    """
    The right operand that evaluates the comparison value. 
    
    This class is responsible for returning the evaluated value
    of the comparison key, i.e., a string, float, etc.
    
    Note that `inf` and `nan` will be evaluated to their
    numpy counterparts. 
    """
    def __init__(self, r):
        super(RightCompOperand, self).__init__(r)
        self.value = self.concat(self.value)
    
    def concat(self, s):
        """
        Concatenate parsed results into one string for eval'ing purposes
        """
        if isinstance(s, six.string_types):
            return s
        else:
            toret = ""
            for x in s: toret += self.concat(x)
            return toret
               
    def eval(self, *args):
        """
        Evaluate the right side of the comparison using ``eval``
        """
        try:
            return eval(self.value, _builtins)
        except Exception as e:
            msg = "right hand side of selection query cannot be eval'ed\n" + "-"*80 + "\n"
            raise SelectionError(msg + "original exception: %s" %str(e))
            
class CompOperator(object):
    """
    Class to parse the comparison operator and
    do the full comparison, joining the LeftCompOperand
    and RightCompOperand instances
    """
    def _is(a, b):
        if numpy.isnan(b):
            return numpy.isnan(a)
        else:
            return operator.eq(a, b)

    def _isnot(a, b):
        if numpy.isnan(b):
            return numpy.logical_not(numpy.isnan(a))
        else:
            return operator.ne(a, b)
    
    ops = {'<' : operator.lt, '<=' : operator.le, 
            '>' : operator.gt, '>=' : operator.ge, 
            '==' : operator.eq, '!=' : operator.ne,
            'is' : _is, 'is not': _isnot}
                            
    def __init__(self, t):
        self.args = t[0][0::2]
        self.reprsymbol = t[0][1]
        
        if len(self.args) != 2 or self.reprsymbol not in self.ops.keys():
            valid = ">=|<=|!=|>|<|==|is|is not"
            raise SelectionError("comparison condition must be two strings separated by %s" %valid)
            
    def __str__(self):
        sep = " %s " % self.reprsymbol
        return "(" + sep.join(map(str,self.args)) + ")"
    
    def eval(self, *data):
        
        # the array of boolean values
        return self.ops[self.reprsymbol](*[a.eval(*data) for a in self.args])

    __repr__ = __str__
    

class BoolBinOp(object):
    def __init__(self,t):
        self.args = t[0][0::2]
    def __str__(self):
        sep = " %s " % self.reprsymbol
        return "(" + sep.join(map(str,self.args)) + ")"
    def eval(self, *data):
        raise NotImplementedError
    __repr__ = __str__
    
class BoolAnd(BoolBinOp):
    reprsymbol = '&'
    def eval(self, *data):
        return numpy.all([a.eval(*data) for a in self.args], axis=0)

class BoolOr(BoolBinOp):
    reprsymbol = '|'
    def eval(self, *data):
        return numpy.any([a.eval(*data) for a in self.args], axis=0)

class BoolNot(object):
    def __init__(self,t):
        self.arg = t[0][1]
    def eval(self, *data):
        return numpy.logical_not(self.arg.eval(*data))
    def __str__(self):
        return "~" + str(self.arg)
    __repr__ = __str__

    
class Query(object):
    """
    Class to parse boolean expressions and return boolean masks
    based on data with named fields. 
    
    Notes
    -----
    *   The string expression must be a `comparison condition`, separated
        by a boolean operator (`and`, `or`, `not`). A comparison condition
        has the syntax: 
            
            `column_name`|`[index]` `comparison_operator` `value`
        
            column_name : 
                the name of a column in a data array. the values from the
                data array with this column name are substituted into
                the boolean expression
    
            comparison_operator :
                any of the following are valid: >, >=, <, <=, ==, !=, is, is not
            
            value : 
                This must be able to have `eval` called on it. Usually
                a number or single-quoted string; nan and inf are also supported.

    *   if `column_name` refers to a vector, the index of the vector can be passed
        as the index of the column, using the usual square bracket syntax
    
    *   `numpy.nan` and `numpy.inf` can be tested for by "is (not) nan" and "is (not) inf"

    *   As many `comparison conditions` as needed can be nested
        together, joined by `and`, `or`, or `not`
    
    *   `log`, `exp`, and `log10` are mapped to their numpy functions
    
    *   any of the above builtin functions can also be applied to the column names, i.e,,
        "log10(Mass) > 14"
    """    
    def __init__(self, str_selection):
        """
        Parameters
        ----------
        str_selection : str
            the boolean expression as a string
        """
        # set up the regex for the individual terms
        operator = pp.Regex(r">=|<=|!=|>|<|==|is not|is")
        number = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?") | pp.Regex(r"(inf|nan)")
        quoted_str = pp.QuotedString("'", unquoteResults=False)
        
        # support for indexing, i.e., `[:,0]`
        integer = pp.Combine( pp.Optional(pp.Literal('-')) + pp.Word(pp.nums) )
        lbracket = pp.Literal("[")
        rbracket = pp.Literal("]")
        index = pp.Optional(lbracket + pp.ZeroOrMore(integer|pp.Literal(":")|pp.Literal(",")) + rbracket)
        index.setParseAction(ColumnSlice)
        
        # left hand side a word, or function on a word
        varname = pp.Word(pp.alphanums, pp.alphanums + "_")
        column_name = pp.Group(varname + index)
        lhs_function_call = pp.Group(varname + pp.Literal("(") + column_name + pp.Literal(")"))
        lhs = lhs_function_call | column_name
        
        # RHS can be arithmetic expression of numbers or a quoted string
        expop = pp.Literal('**')
        multop = pp.oneOf('* /')
        plusop = pp.oneOf('+ -')
        arith_expr = pp.operatorPrecedence(number,
                                [(expop, 2, pp.opAssoc.RIGHT),
                                (multop, 2, pp.opAssoc.LEFT),
                                (plusop, 2, pp.opAssoc.LEFT),]
                                )
        # add support for arithimetic expressions with function calls
        rhs_function_call = pp.Group(varname + pp.Literal("(") + arith_expr + pp.Literal(")"))
        arith_expr_with_functions = pp.operatorPrecedence(number|rhs_function_call,
                                            [(expop, 2, pp.opAssoc.RIGHT),
                                             (multop, 2, pp.opAssoc.LEFT),
                                             (plusop, 2, pp.opAssoc.LEFT),]
                                             )

        rhs = (arith_expr_with_functions|quoted_str)
        condition = pp.Group(lhs + operator + rhs)
        
        # set parsing actions
        lhs.setParseAction(LeftCompOperand)
        rhs.setParseAction(RightCompOperand)
        condition.setParseAction(CompOperator)

        # define expression, based on expression operand and
        # list of operations in precedence order
        self.selection_expr = pp.infixNotation(condition,
                                        [
                                        ("not", 1, pp.opAssoc.RIGHT, BoolNot),
                                        ("and", 2, pp.opAssoc.LEFT,  BoolAnd),
                                        ("or",  2, pp.opAssoc.LEFT,  BoolOr),
                                        ])
                                                          
        # save the string condition and parse it
        self.parse_selection(str_selection)
          
    def __str__(self):
        return self.string_selection
        
    def parse_selection(self, str_selection):
        """
        Parse the input string condition
        """        
        self.string_selection = str_selection
        try:
            self.selection = self.selection_expr.parseString(str_selection)[0]
        except pp.ParseException as e:
            msg = "failure to parse the selection query; see docstring for Query\n" + "-"*80 + "\n"
            raise SelectionError(msg + "original exception: %s" %str(e))
            
        
    def __call__(self, data):
        return self.get_mask(data)

    def get_mask(self, data):
        """
        Apply the selection to the specified data and return the 
        implied boolean mask
        
        Parameters
        ----------
        data : a dict like object
            data object that must have named fields, necessary for
            type-casting the values in the selection string
        
        Returns
        -------
        mask : list or array like
            the boolean mask corresponding to the selection string
        """
        mask = self.selection.eval(data)
        
        # crash if mask selects no objects
        if not mask.sum():
            raise SelectionError("selection query selected no objects")
        return mask
