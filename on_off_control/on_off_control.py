import numpy as np
import ipywidgets as ipw
from hvac.components import Radiator, ControlValve, ControlValveMotor
from hvac.process import Room, OutsideTemperature, DeadTime
from hvac.controller import OnOffController, OutdoorResetController, DataLogger
from nummath import graphing


params = [
    ('<b>Time step</b>', None, None),               # 0
    ('dt', 60, 's'),                                # 1
    ('<b>Radiator</b>', None, None),                # 2
    ('Qe_nom', 1886.0, 'W'),                        # 3
    ('Twe_nom', 75.0, '°C'),                        # 4
    ('Twl_nom', 65.0, '°C'),                        # 5
    ('Tr_nom', 20.0, '°C'),                         # 6
    ('n', 1.3279, None),                            # 7
    ('<b>Room</b>', None, None),                    # 8
    ('R_tr', 0.019, 'K/W'),                         # 9
    ('C_ra', 7.847e4, 'J/K'),                       # 10
    ('C_bm', 1.862e6, 'J/K'),                       # 11
    ('<b>Controller</b>', None, None),              # 12
    ('SP', 22.0, '°C'),                             # 13
    ('HL_offset', 1.0, 'K'),                        # 14
    ('LL_offset', -1.0, 'K'),                       # 15
    ('<b>Outdoor reset line</b>', None, None),      # 16
    ('c0', 61.116, None),                           # 17
    ('c1', -1.568, None),                           # 18
    ('<b>Valve motor</b>', None, None),             # 19
    ('travel_speed', 0.5, '%/s'),                   # 20
    ('<b>Control valve</b>', None, None),           # 21
    ('a', 0.5, None),                               # 22
    ('R', 150, None),                               # 23
    ('V_max', 2.052 / 1000.0 / 60.0, 'm^3/s'),      # 24
    ('<b>Outdoor temperature</b>', None, None),     # 25
    ('T_out_avg', 5.0, '°C'),                       # 26
    ('T_out_ampl', 5.0, 'K'),                       # 27
    ('<b>Measuring delay time</b>', None, None),    # 28
    ('delay', 2, 'number of time steps')            # 29
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
            txt = ipw.HTML(p[0], layout=ipw.Layout(width='200px'))
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
        form_box = ipw.HBox([ipw.VBox(widgets[:16]), ipw.VBox(widgets[16:])])
        form_box.layout.justify_content = "space-around"
        self.check_box = ipw.Checkbox(description='outdoor reset', value=True)
        submit_btn = ipw.Button(description='submit')
        submit_btn.on_click(lambda obj: self.cb_run())
        btn_box = ipw.HBox([self.check_box, submit_btn])
        btn_box.layout.justify_content = "space-around"
        self.plot_area = ipw.Output()
        plot_box = ipw.HBox([self.plot_area])
        plot_box.layout.justify_content = "center"
        gui = ipw.VBox([form_box, btn_box, plot_box])
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

        graph = graphing.MultiGraph(row_num=3, col_num=1, fig_size=[10, 15], dpi=150)

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
