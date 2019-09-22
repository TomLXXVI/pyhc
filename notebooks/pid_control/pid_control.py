import numpy as np
import ipywidgets as ipw
from hvac.components import Radiator, ControlValve, ControlValveMotor
from hvac.process import Room, OutsideTemperature, DeadTime
from hvac.controller import PIDController, OutdoorResetController, DataLogger
from nummath.bq_graphing import LineGraph, GraphMatrix


params = [
    ('<b>Time step</b>', None, None),                                                   # 0
    ('dt', 60, 's'),                                                                    # 1     0
    ('<b>Radiator nominal specifications</b>', None, None),                             # 2
    ('heat output (Q_e)', 1886.0, 'W'),                                                 # 3     1
    ('water entering temperature (T_we)', 75.0, '°C'),                                  # 4     2
    ('water leaving temperature', 65.0, '°C'),                                          # 5     3
    ('room temperature (T_in)', 20.0, '°C'),                                            # 6     4
    ('radiator exponent', 1.3279, None),                                                # 7     5
    ('<b>Room thermal characteristics</b>', None, None),                                # 8
    ('room envelope thermal resistance', 0.019, 'K/W'),                                 # 9     6
    ('room air thermal capacity', 7.847e4, 'J/K'),                                      # 10    7
    ('room envelope effective thermal capacity', 1.862e6, 'J/K'),                       # 11    8
    ('<b>PID controller parameters</b>', None, None),                                   # 12
    ('set point', 22.0, '°C'),                                                          # 13    9
    ('gain factor', -5.0, None),                                                        # 14    10
    ('integral time', 600.0, 's'),                                                      # 15    11
    ('derivative time', 0.0, 's'),                                                      # 16    12
    ('bias', 100.0, '%'),                                                               # 17    13
    ('<b>Outdoor reset line (<code>T_we = c0 + c1 * T_out</code>)</b>', None, None),    # 18
    ('constant term c0', 61.116, '°C'),                                                 # 19    14
    ('slope c1', -1.568, None),                                                         # 20    15
    ('<b>Valve motor</b>', None, None),                                                 # 21
    ('stem travel speed', 0.5, '%/s'),                                                  # 22    16
    ('<b>Control valve</b>', None, None),                                               # 23
    ('valve authority', 0.5, None),                                                     # 24    17
    ('inherent rangeability', 150, None),                                               # 25    18
    ('flow rate at full open position', 2.052 / 1000.0 / 60.0, 'm^3/s'),                # 26    19
    ('<b>Outdoor temperature (T_out)</b>', None, None),                                 # 27
    ('daily average', 5.0, '°C'),                                                       # 28    20
    ('amplitude', 5.0, 'K'),                                                            # 29    21
    ('<b>Measuring sensor</b>', None, None),                                            # 30
    ('delay time', 2, 'number of time steps')                                           # 31    22
]


class ControlLoop:
    def __init__(self, *values):
        self.dt = values[0]
        self.radiator = Radiator(
            Qe_nom=values[1],
            Twe_nom=values[2],
            Twl_nom=values[3],
            Tr_nom=values[4],
            n=values[5]
        )
        self.room = Room(
            R_tr=values[6],
            C_ra=values[7],
            C_bm=values[8],
            dt=self.dt,
            radiator=self.radiator
        )
        self.controller = PIDController(
            SP=values[9],
            Kr=values[10],
            ti=values[11],
            td=values[12],
            bias=values[13],
            dt=self.dt
        )
        self.outdoor_reset_controller = OutdoorResetController(
            c0=values[14],
            c1=values[15]
        )
        self.valve_motor = ControlValveMotor(
            travel_speed=values[16],
            dt=self.dt
        )
        self.control_valve = ControlValve(
            a=values[17],
            R=values[18],
            V_max=values[19]
        )
        self.outside_temperature = OutsideTemperature(
            avg=values[20],
            ampl=values[21],
            period=24*3600.0
        )
        self.dead_time = DeadTime(
            n=int(values[22]),
            init_value=self.outside_temperature.avg
        )

        self.room.initial_values(
            T_r_ini=[self.outside_temperature.avg] * 2,
            T_bm_ini=[self.outside_temperature.avg] * 2
        )
        self.time_axis = np.arange(0.0, 24 * 3600.0 + self.dt, self.dt)
        self.outdoor_reset = True
        self.data_logger = DataLogger('T_out', 'T_we', 'out', 'h', 'V_w', 'T_in', 'T_bm', 'Q_e')

    def run(self):
        num_cycles = 4
        while num_cycles > 0:
            T_in = self.room.T_r[-1]
            for t in self.time_axis:
                if num_cycles == 1:
                    if t == 6 * 3600.0:
                        self.controller.set_point(self.controller.SP - 3.0)
                    if t == 12 * 3600.0:
                        self.controller.set_point(self.controller.SP + 3.0)
                out = self.controller(t, T_in)
                h = self.valve_motor(out)
                V_w = self.control_valve(h)
                T_out = self.outside_temperature(t)
                if self.outdoor_reset is True:
                    T_we = self.outdoor_reset_controller(T_out)
                else:
                    T_we = self.room.radiator.nominal_specs['Twe']
                T_in, T_bm, Q_e = self.room(V_w, T_we, T_out)
                T_in = self.dead_time(T_in)
                if num_cycles == 1:
                    self.data_logger.log(
                        t / 3600.0,
                        T_out=T_out,
                        T_we=T_we,
                        out=out,
                        h=h,
                        V_w=V_w * 6.0e4,
                        T_in=T_in,
                        T_bm=T_bm,
                        Q_e=Q_e
                    )
            num_cycles -= 1

    def get_data(self):
        return self.data_logger.get_data()


class GUI:
    def __init__(self):
        self.control_loop = None
        self.gui = self._create_gui()

    def _create_form(self):
        widgets = []
        self.entries = []
        for p in params:
            txt = ipw.HTML(p[0], layout=ipw.Layout(width='300px'))
            if p[1] is not None:
                entry = ipw.FloatText(value=p[1], layout=ipw.Layout(width='100px'))
                self.entries.append(entry)
            else:
                entry = None
            if p[2] is not None:
                unit = ipw.Label(p[2])
            else:
                unit = None
            if entry and unit:
                widget = ipw.HBox((txt, entry, unit))
            elif entry and not unit:
                widget = ipw.HBox((txt, entry))
            else:
                widget = ipw.HBox((txt,))
            widgets.append(widget)
        return ipw.HBox(
            [ipw.VBox(widgets[:18]), ipw.VBox(widgets[18:])],
            layout=ipw.Layout(justify_content='space-around')
        )

    def _create_button_bar(self):
        self.check_box = ipw.Checkbox(description='outdoor reset active', value=False, indent=False)
        submit_btn = ipw.Button(description='submit')
        submit_btn.on_click(lambda obj: self._cb_run())
        btn_box = ipw.HBox([self.check_box, submit_btn], layout=ipw.Layout(justify_content='space-around'))
        return btn_box

    def _create_diagrams(self):
        values = [entry.value for entry in self.entries]
        self.control_loop = ControlLoop(*values)
        self.control_loop.outdoor_reset = self.check_box.value
        self.control_loop.run()
        return self._build_diagrams()

    def _create_gui(self):
        form = self._create_form()
        button_bar = self._create_button_bar()
        diagrams = self._create_diagrams()
        gui = ipw.VBox([form, button_bar, diagrams])
        gui.layout.justify_content = "flex-start"
        return gui

    def _cb_run(self):
        values = [entry.value for entry in self.entries]
        self.control_loop = ControlLoop(*values)
        self.control_loop.outdoor_reset = self.check_box.value
        self.control_loop.run()
        self._update_diagrams()

    def _build_diagrams(self):
        data = self.control_loop.get_data()
        self.T_graph = LineGraph(
            x_data=data['t'],
            y_data=[data['T_we'], data['T_in'], data['T_bm'], data['T_out']],
            labels=[
                'entering water temperature',
                'indoor temperature',
                'room envelope temperature',
                'outdoor temperature'
            ],
            param_dict={
                'figure': {'title': 'Temperatures'},
                'x-axis': {'title': 'time [h]', 'min': 0, 'max': 24},
                'y-axis': {'title': 'T [°C]', 'min': -20, 'max': 90},
            }
        )
        self.T_graph.add_legend()
        self.T_graph.add_toolbar()
        self.T_graph.draw()
        self.Q_graph = LineGraph(
            x_data=data['t'],
            y_data=[data['Q_e']],
            labels=['radiator heat output'],
            param_dict={
                'figure': {'title': 'Radiator heat output'},
                'x-axis': {'title': 'time [h]', 'min': 0, 'max': 24},
                'y-axis': {'title': 'Qe [W]', 'min': 0, 'max': int(max(data['Q_e']))},
            }
        )
        self.Q_graph.add_toolbar()
        self.Q_graph.draw()
        self.ctrl_graph = LineGraph(
            x_data=data['t'],
            y_data=[data['out'], data['h']],
            labels=['controller output', 'valve stem position'],
            param_dict={
                'figure': {'title': 'Controller output'},
                'x-axis': {'title': 'time [h]', 'min': 0, 'max': 24},
                'y-axis': {'title': 'out, h', 'min': 0, 'max': 100},
            }
        )
        self.ctrl_graph.add_legend()
        self.ctrl_graph.add_toolbar()
        self.ctrl_graph.draw()
        graph_matrix = GraphMatrix(self.T_graph, self.Q_graph, self.ctrl_graph)
        return graph_matrix.get_widget()

    def _update_diagrams(self):
        data = self.control_loop.get_data()
        self.T_graph.update_y_data([data['T_we'], data['T_in'], data['T_bm'], data['T_out']])
        self.Q_graph.update_y_data([data['Q_e']])
        self.Q_graph.y_scale.max = int(max(data['Q_e']))
        self.ctrl_graph.update_y_data([data['out'], data['h']])

    def get_widget(self):
        return self.gui


gui = GUI().get_widget()
