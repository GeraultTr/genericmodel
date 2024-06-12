import inspect as ins
from functools import partial
import multiprocessing as mp


# General process resolution method
class Functor:
    def __init__(self, fun, iteraring: bool = False, total: bool = False):
        self.fun = fun
        self.name = self.fun.__name__[1:]
        self.iterating = iteraring
        self.total = total
        self.input_names = self.inputs(self.fun)
        if len(self.input_names) == 0:
            self.iterating = True

    def inputs(self, fun):
        arguments = ins.getfullargspec(fun)[0]
        arguments.remove("self")
        return arguments

    def __call__(self, chunk=[], *args):
        if self.iterating:
            self.fun(self.instance)
        elif self.data_type == "<class 'dict'>":
            if self.total:
                self.data[self.name].update(
                    {1: self.fun(self.instance, *(getattr(self.instance, arg) for arg in self.input_names))})
            else:
                return {vid: self.fun(self.instance, *(getattr(self.instance, arg)[vid] for arg in self.input_names)) for vid in chunk}
            
        elif self.data_type == "<class 'numpy.ndarray'>":
            self.data[self.name] = self.fun(self.instance, *(self.data[arg] for arg in self.input_names))

# Executor singleton
class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
            class_._instance.priorbalance = {}
            class_._instance.selfbalance = {}
            class_._instance.stepinit = {}
            class_._instance.state = {}
            class_._instance.totalstate = {}
            class_._instance.rate = {}
            class_._instance.totalrate = {}
            class_._instance.deficit = {}
            class_._instance.axial = {}
            class_._instance.potential = {}
            class_._instance.allocation = {}
            class_._instance.actual = {}
            class_._instance.segmentation = {}
            class_._instance.postsegmentation = {}

        return class_._instance


class Choregrapher(Singleton):
    """
    This Singleton class retreives the processes tagged by a decorator in a model class.
    It also provides a __call__ method to schedule model execution.
    """

    scheduled_groups = {}
    iterating_group = {}
    vid_chunks = [[]]
    sub_time_step = {}
    data_structure = {"soil":None, "root":None}
    filter =  {"label": ["Segment", "Apex"], "type":["Base_of_the_root_system", "Normal_root_after_emergence", "Stopped", "Just_Stopped", "Root_nodule"]}

    consensus_scheduling = [
            ["priorbalance", "selfbalance"],
            ["stepinit", "rate", "totalrate", "state", "totalstate"],  # metabolic models
            ["axial"],  # subcategoy for metabolic models
            ["potential", "deficit", "allocation", "actual", "segmentation", "postsegmentation"],  # growth models
        ]

    def add_time_and_data(self, instance, sub_time_step: int, data: dict, compartment: str = "root"):
        module_family = instance.family
        self.sub_time_step[module_family] = sub_time_step
        if self.data_structure[compartment] == None:
            self.data_structure[compartment] = data
            if compartment == "root":
                self.data_structure[compartment]["focus_elements"] = []
        data_structure_type = str(type(list(self.data_structure[compartment].values())[0]))
        for k in self.scheduled_groups[module_family].keys():
            for f_name in self.scheduled_groups[module_family][k].keys():
                self.scheduled_groups[module_family][k][f_name].instance = instance 
                self.scheduled_groups[module_family][k][f_name].data = self.data_structure[compartment]
                self.scheduled_groups[module_family][k][f_name].data_type = data_structure_type

    def add_simulation_time_step(self, simulation_time_step: int):
        """
        Enables to add a global simulation time step to the Choregrapher for it to slice subtimesteps accordingly
        :param simulation_time_step: global simulation time step in seconds
        :return:
        """
        self.simulation_time_step = simulation_time_step

    def add_schedule(self, schedule):
        """
        Method to edit standarded scheduling proposed by the choregrapher. 
        Guidelines :
        - Rows' index in the list are priority order.
        - Elements' index in the rows are in priority order.
        Thus, you should design the priority of this schedule so that "actual rate" comming before "potential state" is indeed the expected behavior in computation scheduling.
        :param schedule: List of lists of stings associated to available decorators :

        For metabolic models, soil models (priority order) : 
        - rate : for process rate computation that will affect model states (ex : transport flow, metabolic consumption) 
        - state : for state balance computation, from previous state and integration of rates' modification (ex : concentrations and contents)
        - deficit : for abnormal state values resulting from rate balance, cumulative deficits are computed before thresholding state values (ex : negative concentrations)

        For growth models (priority order) : 
        - potential : potential element growth computations regarding element initial state
        - actual : actual element growth computations regarding element states actualizing structural states (belongs to state)
        - segmentation : single element partitionning in several uppon actual growth if size exceeds a threshold.
        """
        self.consensus_scheduling = schedule

    def add_process(self, f, name):
        module_family = f.fun.__globals__["family"]
        exists = False
        if module_family not in getattr(self, name).keys():
            getattr(self, name)[module_family] = []
        else:
            for k in range(len(getattr(self, name)[module_family])):
                f_name = getattr(self, name)[module_family][k].name
                if f_name == f.name:
                    getattr(self, name)[module_family][k] = f
                    exists = True
        if not exists:
            getattr(self, name)[module_family].append(f)
        self.build_schedule(module_family=module_family)

    def build_schedule(self, module_family):
        self.scheduled_groups[module_family] = {}
        self.iterating_group[module_family] = {}
        # As functors can belong two multiple categories, we store unique names to avoid duplicated instances
        unique_functors = {}
        for attribute in dir(self):
            if not callable(getattr(self, attribute)) and "_" not in attribute:
                if module_family in getattr(self, attribute).keys():
                    for functor in getattr(self, attribute)[module_family]:
                        if functor.name not in unique_functors.keys():
                            unique_functors[functor.name] = functor
        # Then, We go through these unique functors
        for name, functor in unique_functors.items():
            priority = [0 for k in range(len(self.consensus_scheduling))]
            # We go through each row of the consensus scheduling, in order of priority
            for schedule in range(len(self.consensus_scheduling)):
                # We attribute a number in the functor's tuple to provided decorator.
                for process_type in range(len(self.consensus_scheduling[schedule])):
                    considered_step = getattr(self, self.consensus_scheduling[schedule][process_type])
                    if module_family in considered_step.keys():
                        if name in [f.name for f in considered_step[module_family]]:
                            priority[schedule] = process_type + 1
                            
            # We append the priority tuple to she scheduled groups dictionnary
            if str(priority) not in self.scheduled_groups[module_family].keys():
                self.scheduled_groups[module_family][str(priority)] = {}
                self.iterating_group[module_family][str(priority)] = not functor.iterating and not functor.total and module_family != "soil"
            self.scheduled_groups[module_family][str(priority)][functor.name] = functor

        # Finally, we sort the dictionnary by key so that the call function can go through functor groups in the expected order
        self.scheduled_groups[module_family] = {k: self.scheduled_groups[module_family][k] for k in sorted(self.scheduled_groups[module_family].keys())}

    def __call__(self, module_family):
        if "pool" not in dir(self):
            self.pool = mp.Pool(mp.cpu_count()-1)
        print(module_family)
        for increment in range(int(self.simulation_time_step/self.sub_time_step[module_family])):
            for step in self.scheduled_groups[module_family].keys():
                if self.iterating_group[module_family][step]:
                    
                    map_over_chunks_p = partial(map_over_chunks, self.scheduled_groups[module_family][step])
                    results_chunks = self.pool.map(map_over_chunks_p, self.vid_chunks)

                    for name in results_chunks[0].keys():
                        for results in results_chunks:
                            self.data_structure["root"][name].update(results[name])
                else:
                    for functor in self.scheduled_groups[module_family][step].values():
                        functor(chunk=self.data_structure["root"]["focus_elements"])

        if module_family == "growth":
            for vid in self.data_structure["root"]["struct_mass"].keys():
                if (self.data_structure["root"]["label"][vid] in self.filter["label"] 
                    and self.data_structure["root"]["type"][vid] in self.filter["type"]
                    and vid not in self.data_structure["root"]["focus_elements"]):
                    self.data_structure["root"]["focus_elements"].append(vid)

                    if len(self.vid_chunks[-1]) == 100:
                        self.vid_chunks += [[]]
                    self.vid_chunks[-1].append(vid)


def map_over_chunks(dict_of_functions, chunk):
        results = {}
        for f_name, functor in dict_of_functions.items():
            results[f_name] = functor(chunk)
        return results

# Decorators    
def priorbalance(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, iteraring=True), name="priorbalance")
        return func
    return wrapper()

def selfbalance(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, iteraring=True), name="selfbalance")
        return func
    return wrapper()

def stepinit(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, iteraring=True), name="stepinit")
        return func
    return wrapper()                

def state(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="state")
        return func
    return wrapper()


def rate(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="rate")
        return func
    return wrapper()

def totalrate(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, total=True), name="totalrate")
        return func
    return wrapper()

def deficit(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="deficit")
        return func
    return wrapper()


def totalstate(func):
    def wrapper():
        Choregrapher().add_process(Functor(func, total=True), name="totalstate")
        return func
    return wrapper()

def axial(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="axial")
        return func
    return wrapper()


def potential(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="potential")
        return func
    return wrapper()

def allocation(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="allocation")
        return func
    return wrapper()

def actual(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="actual")
        return func
    return wrapper()


def segmentation(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="segmentation")
        return func
    return wrapper()

def postsegmentation(func):
    def wrapper():
        Choregrapher().add_process(Functor(func), name="postsegmentation")
        return func
    return wrapper()
