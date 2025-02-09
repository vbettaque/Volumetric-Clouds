@tool

class_name CloudNoise
extends Texture3DRD

const SHADER_FILE_PATH: String = "res://noise/cloud_noise.glsl"

const SHADER_LOCAL_SIZE_X: int = 4
const SHADER_LOCAL_SIZE_Y: int = 4
const SHADER_LOCAL_SIZE_Z: int = 4

#enum NoiseType { TYPE_PERLIN, TYPE_WORLEY, TYPE_PERLIN_WORLEY }

@export_range(1, 1024, 1, "or_greater", "suffix:px") var size: int = 128:
	set(new_size):
		size = new_size
		_reformat()
		
@export_range(1, 8, 1) var mipmaps: int = 1:
	set(new_mipmaps):
		mipmaps = new_mipmaps
		_reformat()

func _init():
	RenderingServer.call_on_render_thread(func():
		_init_compute_resources()
		_init_rd_texture()
		_run_compute_process())

func _reformat():
	RenderingServer.call_on_render_thread(func():
		render_device.free_rid.bind(texture_rd_rid)
		_init_rd_texture()
		_run_compute_process())
	emit_changed()

#func _free():
	#RenderingServer.call_on_render_thread(_free_compute_resources)
#
#func _notification(notif):
	#if notif == NOTIFICATION_PREDELETE:
		#_free()

#region Compute Shader Code

var render_device: RenderingDevice

var shader_rid: RID
var pipeline_rid: RID

func _init_compute_resources():
	render_device = RenderingServer.get_rendering_device()
	var shader_file: RDShaderFile = load(SHADER_FILE_PATH)
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()
	shader_rid = render_device.shader_create_from_spirv(shader_spirv)
	pipeline_rid = render_device.compute_pipeline_create(shader_rid)
	
func _init_rd_texture():
	var rd_tf: RDTextureFormat = _create_rd_texture_format()
	texture_rd_rid = render_device.texture_create(rd_tf, RDTextureView.new())
	
func _run_compute_process():
	var uniform_set_rid: RID = _create_uniform_set()
	
	var work_groups_x: int = int((size - 1) / SHADER_LOCAL_SIZE_X) + 1
	var work_groups_y: int = int((size - 1) / SHADER_LOCAL_SIZE_Y) + 1
	var work_groups_z: int = int((size - 1) / SHADER_LOCAL_SIZE_Z) + 1
	
	var compute_list := render_device.compute_list_begin()
	render_device.compute_list_bind_compute_pipeline(compute_list, pipeline_rid)
	render_device.compute_list_bind_uniform_set(compute_list, uniform_set_rid, 0)
	render_device.compute_list_dispatch(compute_list, \
		work_groups_x, work_groups_y, work_groups_z)
	render_device.compute_list_end()

func _create_rd_texture_format() -> RDTextureFormat:
	var rd_tf: RDTextureFormat = RDTextureFormat.new()
	rd_tf.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	rd_tf.texture_type = RenderingDevice.TEXTURE_TYPE_3D
	rd_tf.width = size
	rd_tf.height = size
	rd_tf.depth = size
	rd_tf.mipmaps = mipmaps
	rd_tf.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT \
		+ RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
		+ RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	return rd_tf
	
func _create_uniform_set() -> RID:
	var uniform: RDUniform = RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform.binding = 0
	uniform.add_id(texture_rd_rid)
	return render_device.uniform_set_create([uniform], shader_rid, 0)
	
# How can I call this once the resource freed?
func _free_compute_resources():
	if texture_rd_rid:
		render_device.free_rid(texture_rd_rid)
	if shader_rid:
		render_device.free_rid(shader_rid)

#endregion
