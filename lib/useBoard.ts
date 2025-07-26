import { ref, computed } from 'vue'

export function useBoard() {
  class Point {
    index: [number, number];
    position: [number, number];

    constructor(index: [number, number], position: [number, number]) {
      this.index = index;
      this.position = position;
    }

    public rotate(units: number): Point {
      const angle = units * Math.PI / 3; // Convert units to radians (60 degrees per unit)
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      const newX = this.position[0] * cos - this.position[1] * sin;
      const newY = this.position[0] * sin + this.position[1] * cos;

      return createPointFromPosition([newX, newY]);
    }

    public translate(dx: number, dy: number): Point{
        return createPointFromIndex([this.index[0]+dx, this.index[1]+dy])
    }
  }

  const createPointFromIndex = (idx: [number, number]): Point => {
    const index: [number, number] = [Math.round(idx[0]), Math.round(idx[1])];
    const position: [number, number] = [
      index[0],
      index[1] * Math.cos(Math.PI / 3) // Assuming a simplified hexagonal projection
    ];
    return new Point(index, position);
  };

  const createPointFromPosition = (pos: [number, number]): Point => {
    const position: [number, number] = pos;
    const index: [number, number] = [
      Math.round(position[0]),
      Math.round(position[1] / Math.cos(Math.PI / 3)) // Inverse of the simplified projection
    ];
    return new Point(index, position);
  };

  class Group {
    points: Set<Point>;

    constructor(points: Set<Point>) {
      this.points = points;
    }

    public rotate(units: number): Group {
      const rotatedPoints = this.points.map(point => point.rotate(units));
      return new Group(rotatedPoints);
    }

    public translate(dx: number, dy: number): Group {
      const translatedPoints = this.points.map(point => point.translate(dx, dy));
      return new Group(translatedPoints);
    }
  }

  const createTriangleGroup = (size: number) => {
    let points = new Set<Point>();

    for(let i = 0; i < size; ++i){
        for(let j = 0; j <= i; ++j){
            points.add(createPointFromIndex([-i+2*j, j]))
        }
    }

    return new Group(points);
  }

  class PlayerBoard{
    id: number;
    base: Group;
    pieces: Group | null;

    constructor(id: number){
        this.id = id;
        this.base = createTriangleGroup(4).rotate(3).translate(0, 8).rotate(id);
        this.pieces = null;
    }

    public reset(enabled: boolean){
        if(enabled){
            this.pieces = structuredClone(this.base)
        }
        else{
            this.pieces = null
        }
    }
  }

  class Board{
    base: Group;
    players: Array<PlayerBoard>;

    constructor(){
        let base = new Set<Point>();
        this.players = [];

        for(let i = 0; i < 6; ++i){
            let baseTriangle = createTriangleGroup(5).rotate(i)
            base.union(baseTriangle.points);
            this.players.push(new PlayerBoard(i))
        }

        this.base = new Group(base)
    }
  }

  const board = ref<Board>(new Board())

  return {
    board,
    Point,
    Group,
    PlayerBoard,
    Board,
    createPointFromIndex,
    createPointFromPosition,
    createTriangleGroup,
  }
}
