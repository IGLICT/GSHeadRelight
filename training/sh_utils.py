import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.transform.rotation import Rotation as R
import torch
import numpy as np
from scipy.special import sph_harm

from jaxtyping import Float
from torch import Tensor
from e3nn.o3 import matrix_to_angles, wigner_D
import math
from einops import einsum

# def getSH(N, dirs, basisType='real'):
#     """
#     Get Spherical harmonics up to order N.

#     Parameters:
#     - N: maximum order of harmonics
#     - dirs: [azimuth_1 inclination_1; ...; azimuth_K inclination_K] angles 
#             in radians for each evaluation point, where inclination is the 
#             polar angle from zenith: inclination = pi/2-elevation
#     - basisType: 'complex' or 'real' spherical harmonics

#     Returns:
#     - Y_N: Spherical harmonics values for each direction
#     """
#     azimuth = dirs[:, 0]
#     inclination = dirs[:, 1]
#     num_dirs = dirs.shape[0]
#     num_harmonics = (N + 1) ** 2
#     Y_N = np.zeros((num_dirs, num_harmonics), dtype=np.complex128)

#     index = 0
#     for n in range(N + 1):
#         for m in range(-n, n + 1):
#             if basisType == 'complex':
#                 # Complex spherical harmonics
#                 Y_N[:, index] = sph_harm(m, n, azimuth, inclination)
#             elif basisType == 'real':
#                 # Real spherical harmonics
#                 if m < 0:
#                     Y_N[:, index] = np.sqrt(2) * (-1)**m * sph_harm(-m, n, azimuth, inclination).imag
#                 elif m == 0:
#                     Y_N[:, index] = sph_harm(m, n, azimuth, inclination).real
#                 else:
#                     Y_N[:, index] = np.sqrt(2) * (-1)**m * sph_harm(m, n, azimuth, inclination).real
#             index += 1

#     return Y_N

# def get_shs(imgs, l):
#     # Convert images to double precision
#     imgs = imgs.astype(np.float64)
#     n, h, w, _ = imgs.shape

#     # Compute theta and phi
#     theta = ((np.arange(h) + 0.5) * np.pi / h)
#     phi = ((np.arange(w) + 0.5) * 2 * np.pi / w)

#     # Create meshgrid for directions
#     x, y = np.meshgrid(phi, theta)
#     dirs = np.stack((x.flatten(), y.flatten()), axis=-1)

#     # Compute SH basis values
#     num = (l + 1) ** 2
#     coeff = getSH(l, dirs, 'real')

#     # Compute differential solid angle d(omega)
#     theta = np.arange(h + 1) * np.pi / h
#     val = np.cos(theta)
#     w_theta = val[:-1] - val[1:]
#     val = np.arange(w + 1) * 2 * np.pi / w
#     w_phi = val[1:] - val[:-1]
#     x, y = np.meshgrid(w_phi, w_theta)
#     d_omega = (x * y).flatten()

#     # Compute SH coefficients for all images
#     r = imgs[:, :, :, 0].reshape(n, -1) * d_omega
#     r_coeff = np.sum(r[:, :, np.newaxis] * coeff, axis=1)
#     g = imgs[:, :, :, 1].reshape(n, -1) * d_omega
#     g_coeff = np.sum(g[:, :, np.newaxis] * coeff, axis=1)
#     b = imgs[:, :, :, 2].reshape(n, -1) * d_omega
#     b_coeff = np.sum(b[:, :, np.newaxis] * coeff, axis=1)

#     sh_coeff = np.stack((r_coeff, g_coeff, b_coeff), axis=1)

#     # Compute output images approximated by SH
#     out_r = np.sum(coeff[np.newaxis, :, :] * r_coeff[:, np.newaxis, :], axis=2)
#     out_g = np.sum(coeff[np.newaxis, :, :] * g_coeff[:, np.newaxis, :], axis=2)
#     out_b = np.sum(coeff[np.newaxis, :, :] * b_coeff[:, np.newaxis, :], axis=2)

#     sh_imgs = np.zeros((n, h, w, 3)).astype(np.float32)
#     sh_imgs[:, :, :, 0] = out_r.reshape(n, h, w)
#     sh_imgs[:, :, :, 1] = out_g.reshape(n, h, w)
#     sh_imgs[:, :, :, 2] = out_b.reshape(n, h, w)

#     return sh_coeff, sh_imgs

def getSH(N, dirs, basisType='real'):
    """
    Get Spherical harmonics up to order N.

    Parameters:
    - N: maximum order of harmonics
    - dirs: [azimuth_1 inclination_1; ...; azimuth_K inclination_K] angles 
            in radians for each evaluation point, where inclination is the 
            polar angle from zenith: inclination = pi/2-elevation
    - basisType: 'complex' or 'real' spherical harmonics

    Returns:
    - Y_N: Spherical harmonics values for each direction
    """
    azimuth = dirs[:, 0]
    inclination = dirs[:, 1]
    num_dirs = dirs.shape[0]
    num_harmonics = (N + 1) ** 2
    device = dirs.device
    Y_N = torch.zeros((num_dirs, num_harmonics), dtype=torch.complex128, device=device)

    index = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            if basisType == 'complex':
                # Complex spherical harmonics
                Y_N[:, index] = sph_harm(m, n, azimuth.cpu(), inclination.cpu()).to(device)
            elif basisType == 'real':
                # Real spherical harmonics
                if m < 0:
                    Y_N[:, index] = torch.sqrt(torch.tensor(2.0, device=dirs.device)) * (-1)**m * sph_harm(-m, n, azimuth.cpu(), inclination.cpu()).imag.to(device)
                elif m == 0:
                    Y_N[:, index] = sph_harm(m, n, azimuth.cpu(), inclination.cpu()).real.to(device)
                else:
                    Y_N[:, index] = torch.sqrt(torch.tensor(2.0, device=dirs.device)) * (-1)**m * sph_harm(m, n, azimuth.cpu(), inclination.cpu()).real.to(device)
            index += 1

    return Y_N

def getSH_np(N, dirs, basisType='real'):
    """
    Get Spherical harmonics up to order N.

    Parameters:
    - N: maximum order of harmonics
    - dirs: [azimuth_1 inclination_1; ...; azimuth_K inclination_K] angles 
            in radians for each evaluation point, where inclination is the 
            polar angle from zenith: inclination = pi/2-elevation
    - basisType: 'complex' or 'real' spherical harmonics

    Returns:
    - Y_N: Spherical harmonics values for each direction
    """
    azimuth = dirs[:, 0]
    inclination = dirs[:, 1]
    num_dirs = dirs.shape[0]
    num_harmonics = (N + 1) ** 2
    Y_N = np.zeros((num_dirs, num_harmonics), dtype=np.complex128)

    index = 0
    for n in range(N + 1):
        for m in range(-n, n + 1):
            if basisType == 'complex':
                # Complex spherical harmonics
                Y_N[:, index] = sph_harm(m, n, azimuth, inclination)
            elif basisType == 'real':
                # Real spherical harmonics
                if m < 0:
                    Y_N[:, index] = np.sqrt(2) * (-1)**m * sph_harm(-m, n, azimuth, inclination).imag
                elif m == 0:
                    Y_N[:, index] = sph_harm(m, n, azimuth, inclination).real
                else:
                    Y_N[:, index] = np.sqrt(2) * (-1)**m * sph_harm(m, n, azimuth, inclination).real
            index += 1

    return Y_N

def get_sh(img, l):
    # Convert image to double precision
    img = img.astype(np.float64)
    h, w, _ = img.shape

    # Compute theta and phi
    theta = ((np.arange(h) + 0.5) * np.pi / h)
    phi = ((np.arange(w) + 0.5) * 2 * np.pi / w)

    # Create meshgrid for directions
    x, y = np.meshgrid(phi, theta)
    dirs = np.stack((x.flatten(), y.flatten()), axis=-1)

    # Compute SH basis values
    num = (l + 1) ** 2
    coeff = getSH_np(l, dirs, 'real')

    # Compute differential solid angle d(omega)
    theta = np.arange(h + 1) * np.pi / h
    val = np.cos(theta)
    w_theta = val[:-1] - val[1:]
    val = np.arange(w + 1) * 2 * np.pi / w
    w_phi = val[1:] - val[:-1]
    x, y = np.meshgrid(w_phi, w_theta)
    d_omega = (x * y).flatten()

    # Compute SH coefficients
    r = img[:, :, 0].flatten() * d_omega
    r_coeff = np.sum(r[:, np.newaxis] * coeff, axis=0)
    g = img[:, :, 1].flatten() * d_omega
    g_coeff = np.sum(g[:, np.newaxis] * coeff, axis=0)
    b = img[:, :, 2].flatten() * d_omega
    b_coeff = np.sum(b[:, np.newaxis] * coeff, axis=0)

    sh_coeff = np.stack((r_coeff, g_coeff, b_coeff))

    # Compute output image approximated by SH
    out_r = np.sum(coeff * r_coeff[np.newaxis, :], axis=1)
    out_g = np.sum(coeff * g_coeff[np.newaxis, :], axis=1)
    out_b = np.sum(coeff * b_coeff[np.newaxis, :], axis=1)

    sh_img = np.zeros((h, w, 3)).astype(np.float32)
    sh_img[:, :, 0] = out_r.reshape(h, w)
    sh_img[:, :, 1] = out_g.reshape(h, w)
    sh_img[:, :, 2] = out_b.reshape(h, w)

    return sh_coeff, sh_img

def get_shs(imgs, l, device='cuda'):
    # Convert images to double precision and move to device
    # imgs = torch.tensor(imgs, dtype=torch.float64, device=device)
    n, h, w, _ = imgs.shape

    # Compute theta and phi
    theta = ((torch.arange(h, device=device) + 0.5) * torch.pi / h)
    phi = ((torch.arange(w, device=device) + 0.5) * 2 * torch.pi / w)

    # Create meshgrid for directions
    x, y = torch.meshgrid(phi, theta, indexing='ij')
    dirs = torch.stack((x.flatten(), y.flatten()), dim=-1)

    # Compute SH basis values
    coeff = getSH(l, dirs, 'real')

    # Compute differential solid angle d(omega)
    theta = torch.arange(h + 1, device=device) * torch.pi / h
    val = torch.cos(theta)
    w_theta = val[:-1] - val[1:]
    val = torch.arange(w + 1, device=device) * 2 * torch.pi / w
    w_phi = val[1:] - val[:-1]
    x, y = torch.meshgrid(w_phi, w_theta, indexing='ij')
    d_omega = (x * y).flatten()

    # Compute SH coefficients for all images
    r = imgs[:, :, :, 0].reshape(n, -1) * d_omega
    r_coeff = torch.sum(r[:, :, None] * coeff, dim=1)
    g = imgs[:, :, :, 1].reshape(n, -1) * d_omega
    g_coeff = torch.sum(g[:, :, None] * coeff, dim=1)
    b = imgs[:, :, :, 2].reshape(n, -1) * d_omega
    b_coeff = torch.sum(b[:, :, None] * coeff, dim=1)

    sh_coeff = torch.stack((r_coeff, g_coeff, b_coeff), dim=1)
    return sh_coeff

    # # Compute output images approximated by SH
    # out_r = torch.sum(coeff[None, :, :] * r_coeff[:, None, :], dim=2)
    # out_g = torch.sum(coeff[None, :, :] * g_coeff[:, None, :], dim=2)
    # out_b = torch.sum(coeff[None, :, :] * b_coeff[:, None, :], dim=2)

    # sh_imgs = torch.zeros((n, h, w, 3), dtype=torch.float32, device=device)
    # sh_imgs[:, :, :, 0] = out_r.reshape(n, h, w)
    # sh_imgs[:, :, :, 1] = out_g.reshape(n, h, w)
    # sh_imgs[:, :, :, 2] = out_b.reshape(n, h, w)

    # return sh_coeff, sh_imgs


def rotate_equirectangular_maps(image, rotation_matrices):
    height, width, _ = image.shape
    lon = np.linspace(-np.pi, np.pi, width)
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)
    lon, lat = np.meshgrid(lon, lat)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    directions = np.stack([x, y, z], axis=-1)
    directions = torch.tensor(directions.reshape(-1, 3), dtype=torch.float32)

    # 将 rotation_matrices 转换为 PyTorch 张量
    rotation_matrices = torch.tensor(rotation_matrices, dtype=torch.float32)

    # 批量矩阵乘法
    rotated_directions = torch.bmm(directions.unsqueeze(0).repeat(rotation_matrices.size(0), 1, 1), rotation_matrices.transpose(1, 2))

    # 分解结果
    rotated_x, rotated_y, rotated_z = rotated_directions[..., 0], rotated_directions[..., 1], rotated_directions[..., 2]

    # 计算旋转后的经纬度
    rotated_lon = torch.atan2(rotated_y, rotated_x)
    rotated_lat = torch.asin(rotated_z)

    # 将经纬度转换为像素坐标
    rotated_lon = (rotated_lon + np.pi) / (2 * np.pi) * width
    rotated_lat = (rotated_lat + np.pi / 2) / np.pi * height

    # 将结果转换为 NumPy 数组以便使用 map_coordinates
    rotated_lon = rotated_lon.view(-1, height, width).numpy()
    rotated_lat = rotated_lat.view(-1, height, width).numpy()

    # 初始化结果数组
    rotated_images = np.zeros((rotation_matrices.size(0), height, width, 3), dtype=image.dtype)

    # 使用 map_coordinates 进行插值
    for i in range(3):
        for j in range(rotation_matrices.size(0)):
            rotated_images[j, ..., i] = map_coordinates(image[..., i], [rotated_lat[j], rotated_lon[j]], order=1, mode='nearest')

    return rotated_images

def get_rotation_matrices_to_vectors(dirs, tolerance=1e-5):
    device = dirs.device
    # dir = torch.tensor(dir, dtype=torch.float)
    dirs /= dirs.norm(dim=-1).unsqueeze(-1)
    rotation_matrices = torch.zeros([dirs.shape[0], 3, 3])

    k = torch.tensor([0.,0.,1.]).unsqueeze(0).repeat(dirs.shape[0], 1).to(device)
    i = torch.cross(k, dirs, dim=1)
    j = torch.cross(k, i, dim=1)
    i_norm = i.norm(dim=1).unsqueeze(-1)
    i_norm[i_norm==0] = 1
    j_norm = j.norm(dim=1).unsqueeze(-1)
    j_norm[j_norm==0] = 1
    i /= i_norm
    j /= j_norm

    coord_rot_mats = torch.stack([i,j,k], dim=1)
    new_gammas = torch.acos(torch.mul(k, dirs).sum(dim=1))
    rot_mats_in_new_coord = torch.from_numpy(R.from_euler('x', new_gammas.cpu().numpy()).as_matrix()).float().to(device)
    rotation_matrices = torch.matmul(coord_rot_mats.transpose(1,2), torch.matmul(rot_mats_in_new_coord, coord_rot_mats))
    rotation_matrices = torch.einsum('...ij,...jk->...ik', coord_rot_mats.transpose(1,2), torch.einsum('...ij,...jk->...ik', rot_mats_in_new_coord, coord_rot_mats))
    
    ind1 = ((dirs - k[0]).abs().norm(dim=1) < tolerance)
    ind2 = ((dirs + k[0]).abs().norm(dim=1) < tolerance)
    rotation_matrices[ind1] = torch.eye(3).to(device)
    rotation_matrices[ind2] = torch.diag(torch.tensor([1.,-1.,-1.])).to(device)

    return rotation_matrices

def get_rotation_matrix(dir):
    dir = torch.tensor(dir, dtype=torch.float)
    dir /= dir.norm()

    k = torch.tensor([0.,0.,1.])
    if torch.isclose(k, dir).all():
        return torch.eye(3)
    if torch.isclose(-k, dir).all():
        return torch.diag(torch.tensor([1.,-1.,-1.]))

    i = torch.cross(k, dir)
    j = torch.cross(k, i)
    i /= i.norm()
    j /= j.norm()

    coord_rot_mat = torch.stack([i,j,k])
    new_gamma = torch.acos(torch.dot(k, dir))
    rot_mat_in_new_coord = torch.from_numpy(R.from_euler('x', new_gamma).as_matrix()).float()
    rotation_matrix = torch.matmul(coord_rot_mat.T, torch.matmul(rot_mat_in_new_coord, coord_rot_mat))
    return rotation_matrix

def get_cos_hemisph_envmap():
    h,w=50,100
    envmap = np.zeros([h,w,3])
    for i in range(h//2):
        envmap[i] = (h//2-i) / (h//2)
    return envmap
def recon_imgs(sh_array, l, h=50, w=100, device='cuda'):
    """
    Reconstruct images from multiple sets of spherical harmonics coefficients.

    Parameters:
    - sh_array: array of spherical harmonics coefficients, shape (N, 3, (l+1)^2)
    - l: maximum order of harmonics
    - h: height of the output image
    - w: width of the output image
    - device: device to perform computations on ('cuda' or 'cpu')

    Returns:
    - sh_imgs: array of reconstructed images, shape (N, h, w, 3)
    """
    # Compute theta and phi
    theta = ((torch.arange(h, device=device) + 0.5) * torch.pi / h)
    phi = ((torch.arange(w, device=device) + 0.5) * 2 * torch.pi / w)

    # Create meshgrid for directions
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')
    dirs = torch.stack((phi.flatten(), theta.flatten()), dim=-1)

    # Compute SH basis values
    coeff = getSH(l, dirs, 'real').to(device)  # (h*w, (l+1)^2)

    # Initialize the output tensor
    N = sh_array.shape[0]
    sh_imgs = torch.zeros((N, h, w, 3), dtype=torch.float32, device=device)

    # Convert sh_array to tensor and move to device
    sh_array = torch.tensor(sh_array, dtype=torch.float32, device=device)

    # Reshape coeffs for broadcasting
    coeff = coeff[:, None, None, :]  # (h*w, 1, 1, (l+1)^2)

    # Calculate the reconstructed images for all SH inputs in one go
    for c in range(3):
        sh_imgs[:, :, :, c] = torch.sum(coeff * sh_array[:, None, c, :], dim=-1).reshape(N, h, w)

    return sh_imgs

def recon_img(sh, l, h=50, w=100, device='cuda'):
    if len(sh.shape) == 1:
        sh = sh.unsqueeze(0).repeat(3,1)
    # Compute theta and phi
    theta = ((torch.arange(h, device=device) + 0.5) * torch.pi / h)
    phi = ((torch.arange(w, device=device) + 0.5) * 2 * torch.pi / w)

    # Create meshgrid for directions
    x, y = torch.meshgrid(phi, theta, indexing='ij')
    dirs = torch.stack((x.flatten(), y.flatten()), dim=-1).to(device)

    # Compute SH basis values
    coeff = getSH(l, dirs, 'real').to(device)

    # Convert sh coefficients to tensor and move to device
    # sh = torch.tensor(sh, dtype=torch.float32, device=device)

    # Compute output images approximated by SH
    out_r = torch.sum(coeff * sh[0][None, :], dim=1)
    out_g = torch.sum(coeff * sh[1][None, :], dim=1)
    out_b = torch.sum(coeff * sh[2][None, :], dim=1)

    sh_img = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
    sh_img[:, :, 0] = out_r.reshape(h, w)
    sh_img[:, :, 1] = out_g.reshape(h, w)
    sh_img[:, :, 2] = out_b.reshape(h, w)

    return sh_img

def generate_envmaps(normals, width=100, height=50, device='cuda'):
    # Create an empty tensor for the envmaps
    n = normals.shape[0]
    envmaps = torch.zeros((n, height, width, 3), dtype=torch.float32, device=device)
    
    # Generate theta and phi values for each pixel
    theta = torch.linspace(-torch.pi, torch.pi, width, device=device, dtype=torch.float32)
    phi = torch.linspace(0, torch.pi, height, device=device, dtype=torch.float32)
    
    # Create a meshgrid of theta and phi
    theta, phi = torch.meshgrid(theta, phi, indexing='xy')
    
    # Convert spherical coordinates to Cartesian coordinates
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    # Stack the Cartesian coordinates to form direction vectors
    directions = torch.stack([x, y, z], dim=-1)  # (height, width, 3)
    
    # Normalize the normals tensor to ensure unit vectors
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    
    # Transfer normals to the specified device
    normals = normals.to(device)
    
    # Calculate the cosine of the angle between each normal and the direction vectors
    cos_values = torch.tensordot(directions, normals, dims=([2], [1]))  # (height, width, n)
    cos_values = torch.clamp(cos_values, 0, 1)  # Ensure the values are between 0 and 1
    
    # Transpose the result to get the correct shape
    envmaps = cos_values.permute(2, 0, 1).unsqueeze(-1).repeat(1,1,1,3)  # (n, height, width)
    
    return envmaps


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    # replace to identity matrix if the rotation matrix derminant is not 1 
    # rotations shape is b x 2 x 1 x 1 x 1x 1x 3 x 3
    if not torch.allclose(torch.det(rotations), rotations.new_tensor(1)):
        rotations = torch.eye(3, device=device, dtype=dtype).expand(rotations.shape[:-2] + (3, 3))
    #rotation = torch.eye(3, device=device, dtype=dtype).expand(rotations.shape[:-2] + (3, 3))
    
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(math.floor(math.sqrt(n))):
        # with torch.device(device):
            # sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotations = wigner_D(degree, alpha, beta, gamma, device=device).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)

def clebsch_gordan_coeff():
    # TODO
    return