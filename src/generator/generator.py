
class NotImplementedError(Exception):
    pass

class Generator:
    # TYPE
    def matrix_type(self, rows, cols):
        raise NotImplementedError()
    real_type = None
    integer_type = None
    void_type = None
    # DEFINE
    def define_module(self, name):
        raise NotImplementedError()
    def define_function(self, name, ret_type, params):
        raise NotImplementedError()
    def define_main_function(self):
        raise NotImplementedError()    
    def define_for_block(self, size):
        raise NotImplementedError()
    def define_if_block(self, cond):
        raise NotImplementedError()
    def define_else_if_block(self, cond):
        raise NotImplementedError()
    def define_else_block(self, cond):
        raise NotImplementedError()
    # ADD
    def add_constant(self, name, type, value):
        raise NotImplementedError()
    def add_variable(self, name, type, default_value = None):
        raise NotImplementedError()
    def add_instruction(self, instr):
        raise NotImplementedError()
    # Others
    def param(self, name, type, default_value = None):
        raise NotImplementedError()
    def ref_type(self, type):
        raise NotImplementedError()
    def list_type(self, type):
        raise NotImplementedError()
    def call(self, name, params):
        raise NotImplementedError()
    def assign(self, var, val):
        raise NotImplementedError()
    def matrix_get(self, name, i,j):
        raise NotImplementedError()
    def list_get(self, name, i):
        raise NotImplementedError()


def indent(multiline_string):
    lines = multiline_string.split('\n')
    ret = []
    for l in lines:
        ret.append("  " + l)
    return "\n".join(ret)

class CppMatrixType:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    def __str__(self):
        return "double*"

class CppGenerator(Generator):
    def matrix_type(self, rows, cols):
        return CppMatrixType(rows, cols)
    real_type = "double"
    integer_type = "int"
    void_type = "void"
    def param(self, name, type):
        return "{type} {name}".format(name=name, type=str(type))
    def param_default(self, name, type, default_value):
        return "{type} {name} = {val}".format(
            name=name,
            type=str(type),
            val=str(default_value)
        )
    def ref_type(self, type):
        return "{type}&".format(type=str(type))
    def list_type(self, type):
        return "{type}*".format(type=str(type))
    def call(self, name, params):
        return "{name}({params})".format(
            name=name,
            params=", ".join(map(lambda p: str(p), params))
        )
    def assign(self, var, val):
        return "{var} = {val}".format(
            var=var,
            val=val
        )
    def matrix_get(self, name, type, i,j):
        return "{matrix}[{index}]".format(
            matrix = name,
            index = str(i*type.cols+j),
        )
    def list_get(self, name, i):
        raise NotImplementedError()
    
class CppBlockGenerator(CppGenerator):
    def __init__(self, header):
        self.source = []
        self.header = header
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass
    def add_instruction(self, instr):
        self.source += [instr + ";"]
        return self
    def add_variable(self, name, type, default_value = None):
        self.source += [ self.param_default(name, type, default_value) + ";" ]
        return self
    def define_for_block(self, size):
        template = "for({init} ; {end} ; {inc})"
        header = template.format(
            init = "int i = 0",
            end = "i < " + str(size),
            inc = "i++"
        )
        ret = CppBlockGenerator(header)
        self.source += [ret]
        return ret
    def __str__(self):
        ret  = [self.header + "{"]
        ret += [indent("\n".join(map(lambda i: str(i), self.source)))]
        ret += ["}"]
        return "\n".join(ret)
    
class CppModuleGenerator(CppGenerator):
    def __init__(self, name):
        self.name = name
        self.header = []
        self.source = []
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass
    def add_constant(self, name, type, value):
        template = "#define {name} (({type}){value})"
        self.header += [
            template.format(name=name, type=str(type), value=str(value))
        ]
        return self
    def define_function(self, name, ret_type, params):
        template = "{ret} {name}({params})"
        signature = template.format(
            ret=str(ret_type),
            name=name,
            params=", ".join(params)
        )
        ret = CppBlockGenerator(signature)
        self.header += [signature + ";"]
        self.source += [ret]
        return ret
    def define_main_function(self):
        return self.define_function(
            "main",
            self.integer_type,
            [
                self.param("", self.integer_type),
                self.param("", "char**")
            ]
        )
    def __str__(self):
        ret  = ["// MODULE : " + self.name]
        ret += ["//// HEADER"]
        for i in self.header:
            ret += [str(i)]
        ret += ["//// SOURCE"]
        for i in self.source:
            ret += [str(i)]
        return "\n".join(ret)

class CppProjectGenerator(CppGenerator):
    def __init__(self):
        self.modules = [
            CppModuleGenerator("main")
        ]
    def define_module(self, name):
        ret = CppModuleGenerator(name)
        self.modules += [ret]
        return ret
    def add_constant(self, name, type, value):
        self.modules[0].add_constant(name, type, value)
        return self
    def define_function(self, name, ret_type, params):
        return self.modules[0].define_function(name, ret_type, params)
    def define_main_function(self):
        return self.modules[0].define_main_function()
    def __str__(self):
        ret = ""
        for m in self.modules:
            ret += "////////////////////////////////\n"
            ret += str(m)
            ret += '\n'
        return ret
    

# gen = CppProjectGenerator()
# test = gen.define_module("test")
# test.add_constant("MYCONST", gen.integer_type, 42)
# gen.add_constant("MYCONST", gen.integer_type, 3)
# func = test.define_function("my_func", gen.void_type, [gen.param("my_param", gen.real_type)])
# func.add_instruction(
#     func.assign(
#         "test",
#         func.call("my_func", [0])
#     )
# )
# l = func.define_for_block(10)
# l = l.define_for_block(20)
# gen.define_main_function()
# print(gen)

import sympy
import numpy

def get_cse(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    expr_list = []
    coord_list = {}
    for row in range(0, rows):
        for col in range(0, cols):
            scase = sympy.sympify(matrix[row][col])
            coord_list[(row,col)] = len(expr_list)
            expr_list.append(scase)
    cse = sympy.cse(expr_list, sympy.numbered_symbols('tmp'))
    tmp_list = cse[0]
    expr_list = cse[1]
    expr_mat = []
    for row in range(0, rows):
        row_list = []
        for col in range(0, cols):
            i = coord_list[(row,col)]
            e = expr_list[i]
            row_list.append(expr_list[i])
        expr_mat.append(row_list)
    return (tmp_list, expr_mat)

def get_transposed_jacobian(mat, sym):
    flat = []
    for row in mat:
        for case in row:
            flat.append(case)
    t_jacobian = []
    for s in sym:
        t_jacobian.append(list(map(lambda e: e.diff(s), flat)))
    return t_jacobian

def add_typed_variable_list(gen, vtype, vlist):
    for v in vlist:
        gen.add_variable(v[0], vtype, v[1])
    return gen

def add_fill_matrix(gen, name, matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(0, rows):
        for col in range(0, cols):
            gen.add_instruction(
                gen.assign(
                    gen.matrix_get(
                        name,
                        gen.matrix_type(rows, cols),
                        row, col
                    ),
                    str(matrix[row][col])
                )
            )
    return gen

def add_matrix_returning_function(gen, name, matrix, symbols):
    cse = get_cse(matrix)
    func = gen.define_function(
        name,
        gen.void_type,
        map(lambda s: gen.param(str(s), gen.real_type), symbols) +
        [gen.param("matrix_out", gen.matrix_type(2,2))]
    )
    add_typed_variable_list(func, func.real_type, cse[0])
    add_fill_matrix(func, "matrix_out", cse[1])
    return gen


x = sympy.Symbol("x")
y = sympy.Symbol("y")
m = [[2*x,y+1],[2*x+y+1,y+1]]

gen = CppProjectGenerator()
add_matrix_returning_function(gen, "mymatrix", m, [x,y])
print(gen)
