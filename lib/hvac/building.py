class BuildingElement:
    def __init__(self, **kwargs):
        self._A = kwargs['A'] if 'A' in kwargs.keys() else 1.0           # area [m^2]
        self._t = kwargs['t'] if 't' in kwargs.keys() else 0.0           # thickness [m]

        self._k = kwargs['k'] if 'k' in kwargs.keys() else 0.0           # conduction coefficient [W/(m.K)]
        self._rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.0     # mass density [kg/m^3]
        self._cm = kwargs['cm'] if 'cm' in kwargs.keys() else 0.0        # specific thermal capacity [J/(kg.K)]
        self._cf_r = kwargs['cf_r'] if 'cf_r' in kwargs.keys() else 0.0  # correction factor for resistance

        self._r = kwargs['r'] if 'r' in kwargs.keys() else 0.0           # thermal resistance of 1 m^2 [(m^2.K)/W]
        self._ca = kwargs['ca'] if 'ca' in kwargs.keys() else 0.0        # thermal capacity per m^2 [J/(m^2.K)]

    @property
    def r(self):
        if not self._r:
            self._r = self._t / self._k + self._cf_r
        return self._r

    @property
    def u(self):
        return 1.0 / self.r

    @property
    def U(self):
        return self.u * self._A

    @property
    def ca(self):
        if not self._ca:
            self._ca = self._rho * self._cm * self._t
        return self._ca

    @property
    def C(self):
        return self.ca * self._A

    @property
    def A(self):
        return self._A

    @property
    def t(self):
        return self._t


class BuildingCompositeElement:
    """Building elements in parallel."""
    def __init__(self, *building_elements):
        self._r = 0.0
        self._ca = 0.0
        self.building_elements = building_elements

    @property
    def r(self):
        A = 0.0
        U = 0.0
        for elem in self.building_elements:
            A += elem.A
            U += elem.U
        self._r = A / U
        return self._r

    @property
    def ca(self):
        A = 0.0
        C = 0.0
        for elem in self.building_elements:
            A += elem.A
            C += elem.C
        self._ca = C / A
        return self._ca

    @property
    def t(self):
        for elem in self.building_elements:
            if elem.t: return elem.t
        return 0.0


class BuildingPart:
    """Building elements in series."""
    def __init__(self, *elements, cf_u=0.0):
        self.building_elements = elements

        self._r = 0.0               # specific thermal resistance of building part [(m^2.K)/W]
        self._ca = 0.0              # specific thermal capacity per unit area of building part [J/(m^2.K)]
        self._t = 0.0               # total thickness of building part [m]
        self._cf_u = cf_u           # eventual correction factor for u-value

        self.T_inside = 0.0         # temperature at the inside of the building part [°C]
        self.T_outside = 0.0        # temperature at the outside of the building part [°C]

        self.A_inside = 0.0         # inside surface of building part [m^2]
        self.A_outside = 0.0        # outside surface of building part [m^2]

    @property
    def r(self):
        self._r = 0.0
        for elem in self.building_elements:
            self._r += elem.r
        self._r = 1.0 / (1.0 / self._r + self._cf_u)
        return self._r

    @property
    def u(self):
        return 1.0 / self.r

    @property
    def ca(self):
        self._ca = 0.0
        for elem in self.building_elements:
            self._ca += elem.ca
        return self._ca

    @property
    def t(self):
        self._t = 0.0
        for elem in self.building_elements:
            self._t += elem.t
        return self._t

    def __call__(self, T_inside=0.0, T_outside=0.0, A_inside=0.0, A_outside=0.0):
        self.T_inside = T_inside
        self.T_outside = T_outside
        self.A_inside = A_inside
        self.A_outside = A_outside


class Space:
    def __init__(self, *building_parts):
        self.building_parts = building_parts
        self.T_inside = 22.0        # space air temperature [°C]
        self.T_outside = -10.0      # outside air temperature [°C]
        self.r_conv_wi = 0.13       # thermal convection resistance on inside of wall [(m^2.K)/W]
        self.r_conv_ci = 0.11       # thermal convection resistance on inside of ceiling [(m^2.K)/W]
        self.r_conv_o = 0.04        # thermal convection resistance on the outside [(m^2.K)/W]
        self.floor_area = 0.0       # floor area [m^2]
        self.space_height = 0.0     # height of space [m]
        self.Qtr = 0.0              # transmission heat loss [W]

    def __call__(self, T_inside=22.0, T_outside=-10.0, r_conv_wi=0.13, r_conv_ci=0.11, r_conv_o=0.04, floor_area=0.0,
                 space_height=0.0):
        self.T_inside = T_inside
        self.T_outside = T_outside
        self.r_conv_wi = r_conv_wi
        self.r_conv_ci = r_conv_ci
        self.r_conv_o = r_conv_o
        self.floor_area = floor_area
        self.space_height = space_height

    def _Rtr(self):
        """Global thermal transmission resistance [K/W]."""
        self.Qtr = 0.0  # transmission heat loss of space [W]
        for building_part in self.building_parts:
            self.Qtr += building_part.u * (building_part.A_inside + building_part.A_outside) / 2.0 \
                   * (building_part.T_inside - building_part.T_outside)
        return (self.T_inside - self.T_outside) / self.Qtr

    @property
    def R_rm1(self):
        """Global thermal convection resistance between room air and building mass [K/W]."""
        A_inside = 0.0
        for building_part in self.building_parts:
            A_inside += building_part.A_inside
        return self.r_conv_wi / A_inside

    @property
    def R_rm2(self):
        """Global thermal conduction resistance between room air and building mass [K/W]."""
        return 0.5 * self._Rtr()

    @property
    def R_rm(self):
        """Global thermal resistance (convection + conduction) between room air and building mass [K/W]."""
        return self.R_rm1 + self.R_rm2

    @property
    def R_mo1(self):
        """Global thermal conduction resistance between building mass and outside environment [K/W]."""
        return 0.5 * self._Rtr()

    @property
    def R_mo2(self):
        """Global thermal convection resistance between building mass and outside environment [K/W]."""
        A_outside = 0.0
        for building_part in self.building_parts:
            A_outside += building_part.A_outside
        return self.r_conv_o / A_outside

    @property
    def R_mo(self):
        """Global thermal resistance (convection + conduction) between building mass and outside air [K/W]."""
        return self.R_mo1 + self.R_mo2

    @property
    def C_r(self):
        """Thermal capacity of room air [J/K]."""
        V_inside = self.floor_area * self.space_height
        return 1.205 * 1005.0 * V_inside

    @property
    def C_m(self):
        """"Thermal capacity of building mass [J/K]."""
        C_m = 0.0
        for building_part in self.building_parts:
            C_m += building_part.ca * (building_part.A_inside + building_part.A_outside) / 2.0
        return C_m


if __name__ == '__main__':
    import copy

    facing_brick = BuildingElement(A=1.22e-2, t=0.088, k=0.91, rho=1500.0, cm=840.0)
    facing_mortar = BuildingElement(A=3.18e-3, t=0.088, k=1.5, rho=1800.0, cm=840.0)
    facade = BuildingCompositeElement(facing_brick, facing_mortar)

    cavity = BuildingElement(t=0.02, r=0.085, rho=1.205, cm=1005.0)

    insulation = BuildingElement(t=0.06, k=0.041, rho=25.0, cm=840.0, cf_r=-0.1)

    big_brick = BuildingElement(A=3.97e-2, t=0.138, k=0.32, rho=1100.0, cm=840.0)
    inner_mortar = BuildingElement(A=3.46e-3, t=0.138, k=1.0, rho=1900.0, cm=840.0)
    inner_wall = BuildingCompositeElement(big_brick, inner_mortar)

    plaster = BuildingElement(t=0.01, k=0.57, rho=1300.0, cm=840.0)

    wall = BuildingPart(facade, cavity, insulation, inner_wall, plaster, cf_u=0.018)

    concrete_slab = BuildingElement(t=0.15, k=1.9, rho=2500.0, cm=840.0)

    ceiling = BuildingPart(concrete_slab, plaster)

    floor_surface = BuildingElement(t=0.015, k=3.5, rho=3000.0, cm=840.0)

    floor = BuildingPart(concrete_slab, floor_surface)

    print(f'wall: r = {wall.r} (m^2.K)/W, ca = {wall.ca} J/(m^2.K)')
    print(f'ceiling: r = {ceiling.r} (m^2.K)/W, ca = {ceiling.ca} J/(m^2.K)')
    print(f'floor: r = {floor.r} (m^2.K)/W, ca = {floor.ca} J/(m^2.K)')

    wall_1 = copy.deepcopy(wall)
    wall_2 = copy.deepcopy(wall)
    wall_3 = copy.deepcopy(wall)
    wall_4 = copy.deepcopy(wall)

    wall_1(
        T_inside=22.0,
        T_outside=-10.0,
        A_inside=9.0,
        A_outside=9.0 + wall_2.t + wall_4.t
    )
    wall_2(
        T_inside=22.0,
        T_outside=-10.0,
        A_inside=9.0,
        A_outside=9.0 + wall_1.t + wall_4.t
    )
    wall_3(
        T_inside=22.0,
        T_outside=-10.0,
        A_inside=12.0,
        A_outside=12.0 + wall_2.t + wall_4.t
    )
    wall_4(
        T_inside=22.0,
        T_outside=-10.0,
        A_inside=12.0,
        A_outside=12.0 + wall_1.t + wall_3.t
    )
    ceiling(
        T_inside=22.0,
        T_outside=18.0,
        A_inside=12.0,
        A_outside=(4.0 + wall_1.t + wall_3.t) * (3.0 + wall_2.t + wall_4.t)
    )
    floor(
        T_inside=22.0,
        T_outside=10.0,
        A_inside=12.0,
        A_outside=ceiling.A_outside
    )

    room = Space(wall_1, wall_2, wall_3, wall_4, floor, ceiling)
    room(
        T_inside=22.0,
        T_outside=-10.0,
        floor_area=floor.A_inside,
        space_height=3.0
    )

    print(f"thermal resistance between 'room air' and 'building mass' = {room.R_rm:.2e} K/W")
    print(f"thermal resistance between 'building mass' and 'outside' = {room.R_mo:.2e} K/W")
    print(f"thermal capacity of 'room air' = {room.C_r:.2e} J/K")
    print(f"thermal capacity of 'building mass' = {room.C_m:.2e} J/K")
