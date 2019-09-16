import numpy as np
import ipywidgets as ipw
from hvac.components import Radiator, ControlValve, ControlValveMotor
from hvac.process import Room, OutsideTemperature, DeadTime
from hvac.controller import OnOffController, OutdoorResetController, DataLogger
from nummath import graphing


params = [
    ('<b>Time step</b>', None, None),                                               # 0
    ('dt', 60, 's'),                                                                # 1
    ('<b>Radiator nominal specifications</b>', None, None),                         # 2
    ('heat output', 1886.0, 'W'),                                                   # 3
    ('water entering temperature', 75.0, '°C'),                                     # 4
    ('water leaving temperature', 65.0, '°C'),                                      # 5
    ('room temperature', 20.0, '°C'),                                               # 6
    ('radiator exponent', 1.3279, None),                                            # 7
    ('<b>Room thermal characteristics</b>', None, None),                            # 8
    ('room envelope thermal resistance', 0.019, 'K/W'),                             # 9
    ('room air thermal capacity', 7.847e4, 'J/K'),                                  # 10
    ('room envelope effective thermal capacity', 1.862e6, 'J/K'),                   # 11
    ('<b>On/off-controller parameters</b>', None, None),                            # 12
    ('set point', 22.0, '°C'),                                                      # 13
    ('high dead band limit', 1.0, 'K'),                                             # 14
    ('low dead band limit', -1.0, 'K'),                                             # 15
    ('<b>Outdoor reset line (<code>Twe = c0 + c1 * To</code>)</b>', None, None),    # 16
    ('constant term c0', 61.116, None),                                             # 17
    ('slope c1', -1.568, None),                                                     # 18
    ('<b>Valve motor</b>', None, None),                                             # 19
    ('stem travel speed', 0.5, '%/s'),                                              # 20
    ('<b>Control valve</b>', None, None),                                           # 21
    ('valve authority', 0.5, None),                                                 # 22
    ('inherent rangeability', 150, None),                                           # 23
    ('flow rate at full open position', 2.052 / 1000.0 / 60.0, 'm^3/s'),            # 24
    ('<b>Outdoor temperature</b>', None, None),                                     # 25
    ('daily average', 5.0, '°C'),                                                   # 26
    ('amplitude', 5.0, 'K'),                                                        # 27
    ('<b>Measuring sensor</b>', None, None),                                        # 28
    ('delay time', 2, 'number of time steps')                                       # 29
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
        self.controller = OnOffController(
            SP=values[9],
            HL_offset=values[10],
            LL_offset=values[11],
            dt=self.dt
        )
        self.outdoor_reset_controller = OutdoorResetController(
            c0=values[12],
            c1=values[13]
        )
        self.valve_motor = ControlValveMotor(
            travel_speed=values[14],
            dt=self.dt
        )
        self.control_valve = ControlValve(
            a=values[15],
            R=values[16],
            V_max=values[17]
        )
        self.outside_temperature = OutsideTemperature(
            avg=values[18],
            ampl=values[19],
            period=24*3600.0
        )
        self.dead_time = DeadTime(
            n=int(values[20]),
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
        self.gui = self._create_gui()
        self._init_gui()

    def _create_gui(self):
        widgets = []
        self.entries = []
        for p in params:
            txt = ipw.HTML(p[0], layout=ipw.Layout(width='300px'))
            if p[1]:
                entry = ipw.FloatText(value=p[1], layout=ipw.Layout(width='100px'))
                self.entries.append(entry)
            else:
                entry = None
            if p[2]:
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
        form_box = ipw.VBox(widgets)
        self.check_box = ipw.Checkbox(description='outdoor reset active', value=True, indent=False)
        submit_btn = ipw.Button(description='submit')
        submit_btn.on_click(lambda obj: self.cb_run())
        btn_box = ipw.VBox([self.check_box, submit_btn])
        btn_box.layout.justify_content = "flex-start"
        self.plot_area = ipw.Output()
        plot_box = ipw.HBox([self.plot_area])
        gui = ipw.HBox([ipw.VBox([form_box, btn_box], layout=ipw.Layout(flex='0 0 auto')), plot_box])
        gui.layout.justify_content = "flex-start"
        return gui

    def _init_gui(self):
        values = [entry.value for entry in self.entries]
        self.control_loop = ControlLoop(*values)
        self.control_loop.outdoor_reset = self.check_box.value
        self.control_loop.run()
        with self.plot_area:
            self._plot()

    def cb_run(self):
        self.plot_area.clear_output(wait=True)
        self._init_gui()

    def _plot(self):
        data = self.control_loop.data_logger.get_data()

        graph = graphing.MultiGraph(row_num=3, col_num=1, fig_size=[8, 12], dpi=96)

        graph[1].add_data_set('T_out', data['t'], data['T_out'])
        graph[1].add_data_set('T_bm', data['t'], data['T_bm'])
        graph[1].add_data_set('T_in', data['t'], data['T_in'])
        graph[1].add_data_set('T_we', data['t'], data['T_we'])
        graph[1].add_legend(loc='upper right')
        graph[1].set_axis_titles('time (h)', 'T_out, T_bm, T_in, T_we (°C)')
        graph[1].scale_x_axis(data.iloc[0, 0], data.iloc[-1, 0], 2.0)
        graph[1].scale_y_axis(-10.0, 80.0, 5.0)
        graph[1].turn_grid_on()
        graph[1].draw_graph()

        graph[2].add_data_set('Qe', data['t'], data['Q_e'])
        graph[2].set_axis_titles('time (h)', 'Q_e (W)')
        graph[2].scale_x_axis(data.iloc[0, 0], data.iloc[-1, 0], 2.0)
        graph[2].turn_grid_on()
        graph[2].draw_graph()

        graph[3].add_data_set('out', data['t'], data['out'])
        graph[3].add_data_set('h', data['t'], data['h'])
        graph[3].add_legend()
        graph[3].set_axis_titles('time (h)', 'out, h (%)')
        graph[3].scale_x_axis(data.iloc[0, 0], data.iloc[-1, 0], 2.0)
        graph[3].turn_grid_on()
        graph[3].draw_graph()

        graph.show_graph()

    def show(self):
        return self.gui


gui = GUI().show()
