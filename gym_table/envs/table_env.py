import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_table.utils import *
from enum import IntEnum
import math


class TableEnv(gym.Env):
  """
  2D grid world game environment
  """

  metadata = {
    'render.modes': ['human', 'rgb_array', 'pixmap'],
    'video.frames_per_second': 10
  }

  # Enumeration of possible actions
  class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6

  def __init__(
          self,
          grid_size=None,
          width=None,
          height=None,
          max_steps=100,
          see_through_walls=False,
          seed=1337
  ):
    # Can't set both grid_size and width/height
    if grid_size:
      assert width == None and height == None
      width = grid_size
      height = grid_size

    # Action enumeration for this environment
    self.actions = TableEnv.Actions

    # Actions are discrete integer values
    self.action_space = spaces.Discrete(len(self.actions))

    # Observations are dictionaries containing an
    # encoding of the grid and a textual 'mission' string
    self.observation_space = spaces.Box(
      low=0,
      high=255,
      shape=OBS_ARRAY_SIZE,
      dtype='uint8'
    )
    self.observation_space = spaces.Dict({
      'image': self.observation_space
    })

    # Range of possible rewards
    self.reward_range = (0, 1)

    # Renderer object used to render the whole grid (full-scale)
    self.grid_render = None

    # Renderer used to render observations (small-scale agent view)
    self.obs_render = None

    # Environment configuration
    self.width = width
    self.height = height
    self.max_steps = max_steps
    self.see_through_walls = see_through_walls

    # Starting position and direction for the agent
    self.start_pos = None
    self.start_dir = None

    # Initialize the RNG
    self.seed(seed=seed)

    # Initialize the state
    self.reset()

  def reset(self):
    # Generate a new random grid at the start of each episode
    # To keep the same grid for each episode, call env.seed() with
    # the same seed before calling env.reset()
    self._gen_grid(self.width, self.height)

    # These fields should be defined by _gen_grid
    assert self.start_pos is not None
    assert self.start_dir is not None

    # Check that the agent doesn't overlap with an object
    start_cell = self.grid.get(*self.start_pos)
    assert start_cell is None or start_cell.can_overlap()

    # Place the agent in the starting position and direction
    self.agent_pos = self.start_pos
    self.agent_dir = self.start_dir

    # Item picked up, being carried, initially nothing
    self.carrying = None

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    obs = self.gen_obs()
    return obs

  def seed(self, seed=1337):
    # Seed the random number generator
    self.np_random, _ = seeding.np_random(seed)
    return [seed]

  @property
  def steps_remaining(self):
    return self.max_steps - self.step_count

  def __str__(self):
    """
    Produce a pretty string of the environment's grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
    """

    # Map of object types to short string
    OBJECT_TO_STR = {
      'wall': 'W',
      'floor': 'F',
      'door': 'D',
      'key': 'K',
      'ball': 'A',
      'box': 'B',
      'goal': 'G',
      'lava': 'V'
    }

    # Short string for opened door
    OPENDED_DOOR_IDS = '_'

    # Map agent's direction to short string
    AGENT_DIR_TO_STR = {
      0: '>',
      1: 'V',
      2: '<',
      3: '^'
    }

    str = ''

    for j in range(self.grid.height):

      for i in range(self.grid.width):
        if i == self.agent_pos[0] and j == self.agent_pos[1]:
          str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
          continue

        c = self.grid.get(i, j)

        if c == None:
          str += '  '
          continue

        if c.type == 'door':
          if c.is_open:
            str += '__'
          elif c.is_locked:
            str += 'L' + c.color[0].upper()
          else:
            str += 'D' + c.color[0].upper()
          continue

        str += OBJECT_TO_STR[c.type] + c.color[0].upper()

      if j < self.grid.height - 1:
        str += '\n'

    return str

  def _gen_grid(self, width, height):
    assert False, "_gen_grid needs to be implemented by each environment"

  def _reward(self):
    """
    Compute the reward to be given upon success
    """

    return 1 - 0.9 * (self.step_count / self.max_steps)

  def _rand_int(self, low, high):
    """
    Generate random integer in [low,high[
    """

    return self.np_random.randint(low, high)

  def _rand_float(self, low, high):
    """
    Generate random float in [low,high[
    """

    return self.np_random.uniform(low, high)

  def _rand_bool(self):
    """
    Generate random boolean value
    """

    return (self.np_random.randint(0, 2) == 0)

  def _rand_elem(self, iterable):
    """
    Pick a random element in a list
    """

    lst = list(iterable)
    idx = self._rand_int(0, len(lst))
    return lst[idx]

  def _rand_subset(self, iterable, num_elems):
    """
    Sample a random subset of distinct elements of a list
    """

    lst = list(iterable)
    assert num_elems <= len(lst)

    out = []

    while len(out) < num_elems:
      elem = self._rand_elem(lst)
      lst.remove(elem)
      out.append(elem)

    return out

  def _rand_color(self):
    """
    Generate a random color name (string)
    """

    return self._rand_elem(COLOR_NAMES)

  def _rand_pos(self, xLow, xHigh, yLow, yHigh):
    """
    Generate a random (x,y) position tuple
    """

    return (
      self.np_random.randint(xLow, xHigh),
      self.np_random.randint(yLow, yHigh)
    )

  def place_obj(self,
                obj,
                top=None,
                size=None,
                reject_fn=None,
                max_tries=math.inf
                ):
    """
    Place an object at an empty position in the grid

    :param top: top-left position of the rectangle where to place
    :param size: size of the rectangle where to place
    :param reject_fn: function to filter out potential positions
    """

    if top is None:
      top = (0, 0)

    if size is None:
      size = (self.grid.width, self.grid.height)

    num_tries = 0

    while True:
      # This is to handle with rare cases where rejection sampling
      # gets stuck in an infinite loop
      if num_tries > max_tries:
        raise RecursionError('rejection sampling failed in place_obj')

      num_tries += 1

      pos = np.array((
        self._rand_int(top[0], top[0] + size[0]),
        self._rand_int(top[1], top[1] + size[1])
      ))

      # Don't place the object on top of another object
      if self.grid.get(*pos) != None:
        continue

      # Don't place the object where the agent is
      if np.array_equal(pos, self.start_pos):
        continue

      # Check if there is a filtering criterion
      if reject_fn and reject_fn(self, pos):
        continue

      break

    self.grid.set(*pos, obj)
    if obj is not None:
      obj.init_pos = pos
      obj.cur_pos = pos

    return pos

  def place_agent(
          self,
          top=None,
          size=None,
          rand_dir=True,
          max_tries=math.inf
  ):
    """
    Set the agent's starting point at an empty position in the grid
    """

    self.start_pos = None
    pos = self.place_obj(None, top, size, max_tries=max_tries)
    self.start_pos = pos

    if rand_dir:
      self.start_dir = self._rand_int(0, 4)

    return pos

  @property
  def dir_vec(self):
    """
    Get the direction vector for the agent, pointing in the direction
    of forward movement.
    """

    assert self.agent_dir >= 0 and self.agent_dir < 4
    return DIR_TO_VEC[self.agent_dir]

  @property
  def right_vec(self):
    """
    Get the vector pointing to the right of the agent.
    """

    dx, dy = self.dir_vec
    return np.array((-dy, dx))

  @property
  def front_pos(self):
    """
    Get the position of the cell that is right in front of the agent
    """

    return self.agent_pos + self.dir_vec

  def get_view_coords(self, i, j):
    """
    Translate and rotate absolute grid coordinates (i, j) into the
    agent's partially observable view (sub-grid). Note that the resulting
    coordinates may be negative or outside of the agent's view size.
    """

    ax, ay = self.agent_pos
    dx, dy = self.dir_vec
    rx, ry = self.right_vec

    # Compute the absolute coordinates of the top-left view corner
    sz = AGENT_VIEW_SIZE
    hs = AGENT_VIEW_SIZE // 2
    tx = ax + (dx * (sz - 1)) - (rx * hs)
    ty = ay + (dy * (sz - 1)) - (ry * hs)

    lx = i - tx
    ly = j - ty

    # Project the coordinates of the object relative to the top-left
    # corner onto the agent's own coordinate system
    vx = (rx * lx + ry * ly)
    vy = -(dx * lx + dy * ly)

    return vx, vy

  def get_view_exts(self):
    """
    Get the extents of the square set of tiles visible to the agent
    Note: the bottom extent indices are not included in the set
    """

    # Facing right
    if self.agent_dir == 0:
      topX = self.agent_pos[0]
      topY = self.agent_pos[1] - AGENT_VIEW_SIZE // 2
    # Facing down
    elif self.agent_dir == 1:
      topX = self.agent_pos[0] - AGENT_VIEW_SIZE // 2
      topY = self.agent_pos[1]
    # Facing left
    elif self.agent_dir == 2:
      topX = self.agent_pos[0] - AGENT_VIEW_SIZE + 1
      topY = self.agent_pos[1] - AGENT_VIEW_SIZE // 2
    # Facing up
    elif self.agent_dir == 3:
      topX = self.agent_pos[0] - AGENT_VIEW_SIZE // 2
      topY = self.agent_pos[1] - AGENT_VIEW_SIZE + 1
    else:
      assert False, "invalid agent direction"

    botX = topX + AGENT_VIEW_SIZE
    botY = topY + AGENT_VIEW_SIZE

    return (topX, topY, botX, botY)

  def relative_coords(self, x, y):
    """
    Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
    """

    vx, vy = self.get_view_coords(x, y)

    if vx < 0 or vy < 0 or vx >= AGENT_VIEW_SIZE or vy >= AGENT_VIEW_SIZE:
      return None

    return vx, vy

  def in_view(self, x, y):
    """
    check if a grid position is visible to the agent
    """

    return self.relative_coords(x, y) is not None

  def agent_sees(self, x, y):
    """
    Check if a non-empty grid position is visible to the agent
    """

    coordinates = self.relative_coords(x, y)
    if coordinates is None:
      return False
    vx, vy = coordinates

    obs = self.gen_obs()
    obs_grid = Grid.decode(obs['image'])
    obs_cell = obs_grid.get(vx, vy)
    world_cell = self.grid.get(x, y)

    return obs_cell is not None and obs_cell.type == world_cell.type

  def step(self, action):
    self.step_count += 1

    reward = 0
    done = False

    # Get the position in front of the agent
    fwd_pos = self.front_pos

    # Get the contents of the cell in front of the agent
    fwd_cell = self.grid.get(*fwd_pos)

    # Rotate left
    if action == self.actions.left:
      self.agent_dir -= 1
      if self.agent_dir < 0:
        self.agent_dir += 4

    # Rotate right
    elif action == self.actions.right:
      self.agent_dir = (self.agent_dir + 1) % 4

    # Move forward
    elif action == self.actions.forward:
      if fwd_cell == None or fwd_cell.can_overlap():
        self.agent_pos = fwd_pos
      if fwd_cell != None and fwd_cell.type == 'goal':
        done = True
        reward = self._reward()
      if fwd_cell != None and fwd_cell.type == 'lava':
        done = True

    # Pick up an object
    elif action == self.actions.pickup:
      if fwd_cell and fwd_cell.can_pickup():
        if self.carrying is None:
          self.carrying = fwd_cell
          self.carrying.cur_pos = np.array([-1, -1])
          self.grid.set(*fwd_pos, None)

    # Drop an object
    elif action == self.actions.drop:
      if not fwd_cell and self.carrying:
        self.grid.set(*fwd_pos, self.carrying)
        self.carrying.cur_pos = fwd_pos
        self.carrying = None

    # Toggle/activate an object
    elif action == self.actions.toggle:
      if fwd_cell:
        fwd_cell.toggle(self, fwd_pos)

    # Done action (not used by default)
    elif action == self.actions.done:
      pass

    else:
      assert False, "unknown action"

    if self.step_count >= self.max_steps:
      done = True

    obs = self.gen_obs()

    return obs, reward, done, {}

  def gen_obs_grid(self):
    """
    Generate the sub-grid observed by the agent.
    This method also outputs a visibility mask telling us which grid
    cells the agent can actually see.
    """

    topX, topY, botX, botY = self.get_view_exts()

    grid = self.grid.slice(topX, topY, AGENT_VIEW_SIZE, AGENT_VIEW_SIZE)

    for i in range(self.agent_dir + 1):
      grid = grid.rotate_left()

    # Process occluders and visibility
    # Note that this incurs some performance cost
    if not self.see_through_walls:
      vis_mask = grid.process_vis(agent_pos=(AGENT_VIEW_SIZE // 2, AGENT_VIEW_SIZE - 1))
    else:
      vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

    # Make it so the agent sees what it's carrying
    # We do this by placing the carried object at the agent's position
    # in the agent's partially observable view
    agent_pos = grid.width // 2, grid.height - 1
    if self.carrying:
      grid.set(*agent_pos, self.carrying)
    else:
      grid.set(*agent_pos, None)

    return grid, vis_mask

  def gen_obs(self):
    """
    Generate the agent's view (partially observable, low-resolution encoding)
    """

    grid, vis_mask = self.gen_obs_grid()

    # Encode the partially observable view into a numpy array
    image = grid.encode(vis_mask)

    assert hasattr(self, 'mission'), "environments must define a textual mission string"

    # Observations are dictionaries containing:
    # - an image (partially observable view of the environment)
    # - the agent's direction/orientation (acting as a compass)
    # - a textual mission string (instructions for the agent)
    obs = {
      'image': image,
      'direction': self.agent_dir,
      'mission': self.mission
    }

    return obs

  def get_obs_render(self, obs, tile_pixels=CELL_PIXELS // 2):
    """
    Render an agent observation for visualization
    """

    if self.obs_render == None:
      from gym_minigrid.rendering import Renderer
      self.obs_render = Renderer(
        AGENT_VIEW_SIZE * tile_pixels,
        AGENT_VIEW_SIZE * tile_pixels
      )

    r = self.obs_render

    r.beginFrame()

    grid = Grid.decode(obs)

    # Render the whole grid
    grid.render(r, tile_pixels)

    # Draw the agent
    ratio = tile_pixels / CELL_PIXELS
    r.push()
    r.scale(ratio, ratio)
    r.translate(
      CELL_PIXELS * (0.5 + AGENT_VIEW_SIZE // 2),
      CELL_PIXELS * (AGENT_VIEW_SIZE - 0.5)
    )
    r.rotate(3 * 90)
    r.setLineColor(255, 0, 0)
    r.setColor(255, 0, 0)
    r.drawPolygon([
      (-12, 10),
      (12, 0),
      (-12, -10)
    ])
    r.pop()

    r.endFrame()

    return r.getPixmap()

  def render(self, mode='human', close=False):
    """
    Render the whole-grid human view
    """

    if close:
      if self.grid_render:
        self.grid_render.close()
      return

    if self.grid_render is None:
      from gym_table.rendering import Renderer
      self.grid_render = Renderer(
        self.width * CELL_PIXELS,
        self.height * CELL_PIXELS,
        True if mode == 'human' else False
      )

    r = self.grid_render

    if r.window:
      r.window.setText(self.mission)

    r.beginFrame()

    # Render the whole grid
    self.grid.render(r, CELL_PIXELS)

    # Draw the agent
    r.push()
    r.translate(
      CELL_PIXELS * (self.agent_pos[0] + 0.5),
      CELL_PIXELS * (self.agent_pos[1] + 0.5)
    )
    r.rotate(self.agent_dir * 90)
    r.setLineColor(255, 0, 0)
    r.setColor(255, 0, 0)
    r.drawPolygon([
      (-12, 10),
      (12, 0),
      (-12, -10)
    ])
    r.pop()

    # Compute which cells are visible to the agent
    _, vis_mask = self.gen_obs_grid()

    # Compute the absolute coordinates of the bottom-left corner
    # of the agent's view area
    f_vec = self.dir_vec
    r_vec = self.right_vec
    top_left = self.agent_pos + f_vec * (AGENT_VIEW_SIZE - 1) - r_vec * (AGENT_VIEW_SIZE // 2)

    # For each cell in the visibility mask
    for vis_j in range(0, AGENT_VIEW_SIZE):
      for vis_i in range(0, AGENT_VIEW_SIZE):
        # If this cell is not visible, don't highlight it
        if not vis_mask[vis_i, vis_j]:
          continue

        # Compute the world coordinates of this cell
        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

        # Highlight the cell
        r.fillRect(
          abs_i * CELL_PIXELS,
          abs_j * CELL_PIXELS,
          CELL_PIXELS,
          CELL_PIXELS,
          255, 255, 255, 75
        )

    r.endFrame()

    if mode == 'rgb_array':
      return r.getArray()
    elif mode == 'pixmap':
      return r.getPixmap()

    return r
