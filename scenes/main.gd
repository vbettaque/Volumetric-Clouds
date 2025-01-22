@tool
extends Node3D

@onready var world: WorldEnvironment = $WorldEnvironment

var time: float = 0.
var ticker: int = 0;

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	ticker = (ticker + 1) % 1
	time += delta
	if ticker == 0:
		((world.environment.sky as Sky).sky_material as ShaderMaterial).set_shader_parameter("time", time)
