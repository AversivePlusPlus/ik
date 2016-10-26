
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
    def const_type(self, type):
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
    def nested(self, *names):
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
        return "double[{r}*{c}]".format(
            r=self.rows,
            c=self.cols
        )

class CppGenerator(Generator):
    def matrix_type(self, rows, cols):
        return CppMatrixType(rows, cols)
    real_type = "double"
    integer_type = "int"
    void_type = "void"
    def param(self, name, type, default_value = None):
        if default_value == None:
            return "{type} {name}".format(
                name=name,
                type=str(type),
            )
        return "{type} {name} = {val}".format(
            name=name,
            type=str(type),
            val=str(default_value)
        )
    def ref_type(self, type):
        return "{type}&".format(type=str(type))
    def const_type(self, type):
        return "const {type}".format(type=str(type))
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
    def nested(self, *names):
        return "::".join(names)
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
        self.source += [ self.param(name, type, default_value) + ";" ]
        return self
    def add_constant(self, name, type, value):
        return self.add_variable(name, self.const_type(type), value)
    def define_if_block(self, cond):
        header = "if({})".format(cond)
        ret = CppBlockGenerator(header)
        self.source += [ret]
        return ret
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
        ret  = [self.header + " {"]
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
    def add_instruction(self, instr):
        self.header += [instr + ";"]
        return self
    def define_function(self, name, ret_type, params):
        template = "{ret} {name}({params})"
        header_signature = template.format(
            ret=str(ret_type),
            name=name,
            params=", ".join(params)
        )
        signature = template.format(
            ret=str(ret_type),
            name=self.nested(self.name, name),
            params=", ".join(params)
        )
        ret = CppBlockGenerator(signature)
        self.header += [header_signature + ";"]
        self.source += ["", ret]
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
        if self.header != []:
            ret += ["//// HEADER"]
            ret += ["#ifndef IK_{}_HPP".format(self.name.upper())]
            ret += ["#define IK_{}_HPP".format(self.name.upper()), ""]
            ret += ["namespace {} {{".format(self.name), ""]
            for i in self.header:
                ret += [str(i)]
            ret += ["", "}", ""]
            ret += ["#endif//IK_{}_HPP".format(self.name.upper())]
            ret += [""]
        if self.source != []:
            ret += ["//// SOURCE"]
            ret += ["//#include <{}.hpp>".format(self.name)]
            ret += ["#include <math.h>"]
            for i in self.source:
                ret += [str(i)]
            ret += [""]
        return "\n".join(ret)
    def __iter__(self):
        if self.header != []:
            yield self.name + ".hpp"
        if self.source != []:
            yield self.name + ".cpp"
    def __getitem__(self, fname):
        ret = []
        if fname == self.name + ".hpp":
            ret += ["#ifndef IK_{}_HPP".format(self.name.upper())]
            ret += ["#define IK_{}_HPP".format(self.name.upper()), ""]
            ret += ["namespace {} {{".format(self.name), ""]
            for i in self.header:
                ret += [str(i)]
            ret += ["", "}", ""]
            ret += ["#endif//IK_{}_HPP".format(self.name.upper())]
            ret += [""]
        elif fname == self.name + ".cpp":
            ret += ["#include <{}.hpp>".format(self.name)]
            ret += ["#include <math.h>"]
            for i in self.source:
                ret += [str(i)]
            ret += [""]
        else:
            raise Exception()
        return "\n".join(ret)

class CppProjectGenerator(CppModuleGenerator):
    def __init__(self):
        CppModuleGenerator.__init__(self, "main")
        self.modules = []
    def define_module(self, name):
        ret = CppModuleGenerator(name)
        self.modules += [ret]
        return ret
    def __str__(self):
        ret = ""
        ret += CppModuleGenerator.__str__(self)
        ret += '\n'
        for m in self.modules:
            ret += "////////////////////////////////\n"
            ret += str(m)
            ret += '\n'
        return ret
    def __iter__(self):
        for e in CppModuleGenerator.__iter__(self):
            yield e
        for m in self.modules:
            for e in m:
                yield e
    def __getitem__(self, fname):
        for e in CppModuleGenerator.__iter__(self):
            if e == fname:
                return CppModuleGenerator.__getitem__(self, fname)
        for m in self.modules:
            for e in m:
                if e == fname:
                    return m[e]
            

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

def add_matrix_typedef(gen):
    gen.add_instruction("template<int r, int c> using matrix = {}".format(
        gen.matrix_type("r", "c")
    ))
    return gen

class IkMatrixType:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    def __str__(self):
        return "matrix<{r}, {c}>".format(
            r=self.rows,
            c=self.cols
        )

def matrix_type(r, c):
    return IkMatrixType(r,c)

def add_typed_variable_list(gen, vtype, vlist):
    for v in vlist:
        gen.add_variable(v[0], vtype, v[1])
    return gen

def add_fill_matrix_instructions(gen, name, matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(0, rows):
        for col in range(0, cols):
            gen.add_instruction(
                gen.assign(
                    gen.matrix_get(
                        name,
                        matrix_type(rows, cols),
                        row, col
                    ),
                    str(matrix[row][col])
                )
            )
    return gen

def add_matrix_returning_function(gen, name, matrix, symbols):
    cse = get_cse(matrix)
    rows = len(matrix)
    cols = len(matrix[0])
    func = gen.define_function(
        name,
        gen.void_type,
        map(lambda s: gen.param(str(s), gen.real_type), symbols) +
        [gen.param("matrix_out", gen.ref_type(matrix_type(rows, cols)))]
    )
    add_typed_variable_list(func, func.real_type, cse[0])
    add_fill_matrix_instructions(func, "matrix_out", cse[1])
    return gen

def add_forward_kinematics_function(gen, chain):
    ids = range(0, chain.get_num_params())
    sym = list(map(lambda i: sympy.Symbol("q"+str(i)), ids))
    mat = chain.forward_kinematics(sym)
    add_matrix_returning_function(
        gen,
        "forward_kinematics",
        mat,
        sym
    )
    return gen

def add_transposed_jacobian_function(gen, chain):
    ids = range(0, chain.get_num_params())
    sym = list(map(lambda i: sympy.Symbol("q"+str(i)), ids))
    mat = chain.forward_kinematics(sym)
    t_jacobian = get_transposed_jacobian(mat, sym)
    add_matrix_returning_function(
        gen,
        "transposed_jacobian",
        t_jacobian,
        sym
    )
    return gen

def get_chain_symbols(chain):
    ids = range(0, chain.get_num_params())
    pnames = list(map(lambda i: "q"+str(i)+"_io", ids))
    return list(map(lambda i: sympy.Symbol(i), pnames))

def add_chain_distance_instructions(gen, chain_name, ret_name, target_name, sym, rows, cols):
    gen.add_variable(ret_name, matrix_type(rows, cols))
    gen.add_instruction(
        gen.call(
            "forward_kinematics",
            map(lambda s: str(s), sym) + ["dist"]
        )
    )
    for row in range(0, rows):
        for col in range(0, cols):
            gen.add_instruction(
                gen.assign(
                    gen.matrix_get(
                        ret_name,
                        matrix_type(rows, cols),
                        row, col
                    ),
                    " - ".join([
                        gen.matrix_get(
                            ret_name,
                            matrix_type(rows, cols),
                            row, col
                        ),
                        gen.matrix_get(
                            target_name,
                            matrix_type(rows, cols),
                            row, col
                        ),
                    ])
                )
            )

def get_matrix_mult(m1, m2):
    ret = sympy.Matrix(m1) * sympy.Matrix(m2)
    return list(map(lambda e: [e], list(ret)))

def get_matrix_from_name(gen, name, rows, cols):
    ret = []
    for row in range(0, rows):
        ret_row = []
        for col in range(0, cols):
            ret_row.append(
                sympy.Symbol(gen.matrix_get(name, matrix_type(rows, cols), row, col))
            )
        ret.append(ret_row)
    return ret

def get_column_matrix(matrix):
    ret = []
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(0, rows):
        for col in range(0, cols):
            ret.append([matrix[row][col]])
    return ret

def get_flat_matrix(matrix):
    ret = []
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(0, rows):
        for col in range(0, cols):
            ret.append(matrix[row][col])
    return ret

def add_norm_constant(gen, ret_name, dist_name, dist_rows, dist_cols):
    dist = get_flat_matrix(
        get_matrix_from_name(gen, dist_name, dist_rows, dist_cols)
    )
    gen.add_constant(
        ret_name,
        gen.real_type,
        gen.call(
            "sqrt",
            [" + ".join(map(lambda e: "{e}*{e}".format(e=str(e)), dist))]
        )
    )
    return gen

def add_ik_return_instructions(gen, sym, delta, delta_norm, dist_norm):
    delta = get_flat_matrix(get_matrix_from_name(gen, delta, len(sym), 1))
    ifblock = gen.define_if_block("{} != 0".format(delta_norm))
    ifblock.add_constant(
        "gain",
        ifblock.real_type,
        "{}*{}/{}".format("coeff", dist_norm, delta_norm)
    )
    for (s,d) in zip(sym,delta):
        ifblock.add_instruction(
            ifblock.assign(
                s,
                "{}+({}*{})".format(s, d, "gain")
            )
        )

def add_inverse_kinematics_step_function(gen, chain):
    ids = range(0, chain.get_num_params())
    sym = list(map(lambda i: sympy.Symbol("q"+str(i)+"_io"), ids))
    mat = chain.forward_kinematics(sym)
    shape = sympy.Matrix(mat).shape
    rows = shape[0]
    cols = shape[1]
    t_jacobian = get_transposed_jacobian(mat, sym)
    dist = get_matrix_from_name(gen, "dist", rows, cols)
    delta = get_matrix_mult(t_jacobian, get_column_matrix(dist))
    assert(len(delta) == len(sym))
    assert(len(delta[0]) == 1)
    func = gen.define_function(
        "inverse_kinematics_step",
        gen.void_type,
        [gen.param("target", gen.const_type(gen.ref_type(matrix_type(rows, cols))))] +
        map(lambda s: gen.param(str(s), gen.ref_type(gen.real_type)), sym) +
        [gen.param("coeff", gen.real_type)]
    )
    add_chain_distance_instructions(func, chain.name, "dist", "target", sym, rows, cols)
    func.add_variable("delta", matrix_type(len(sym),1))
    add_fill_matrix_instructions(func, "delta", delta)
    add_norm_constant(func, "delta_norm", "delta", len(sym), 1)
    add_norm_constant(func, "dist_norm", "dist", rows, cols)
    add_ik_return_instructions(func, sym, "delta", "delta_norm", "dist_norm")

def add_ik_module(gen, chain):
    mod = gen.define_module(chain.name)
    add_matrix_typedef(mod)
    add_forward_kinematics_function(mod, chain)
    add_inverse_kinematics_step_function(mod, chain)
    return gen
