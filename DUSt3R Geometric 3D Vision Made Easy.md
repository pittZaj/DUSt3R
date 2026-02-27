**DUSt3R: Geometric 3D Vision Made Easy**



**Abstract**

*Multi-view stereo reconstruction (MVS) in the wild re*

*quires to first estimate the camera intrinsic and extrinsic*

*parameters. These are usually tedious and cumbersome to*

*obtain, yet they are mandatory to triangulate corresponding*

*pixels in 3D space, which is at the core of all best perform*

*ing MVS algorithms. In this work, we take an opposite*

*stance and introduce* **DUSt3R***, a radically novel paradigm*

*for* *D**ense and* *U**nconstrained* *S**tereo* *3**D* *R**econstruction of*

*arbitrary image collections, operating without prior infor*

*mation about camera calibration nor viewpoint poses. We*

*cast the pairwise reconstruction problem as a regression of*

*pointmaps, relaxing the hard constraints of usual projective*

*camera models. We show that this formulation smoothly*

*unifies the monocular and binocular reconstruction cases.*

*In the case where more than two images are provided, we fur*

*ther propose a simple yet effective global alignment strategy*

*that expresses all pairwise pointmaps in a common refer*

*ence frame. We base our network architecture on standard*

*Transformer encoders and decoders, allowing us to leverage*

*powerful pretrained models. Our formulation directly pro*

*vides a 3D model of the scene as well as depth information,*

*but interestingly, we can seamlessly recover from it, pixel*

*matches, focal lengths, relative and absolute cameras. Exten*

*sive experiments on all these tasks showcase how DUSt3R*

*effectively unifies various 3D vision tasks, setting new perfor*

*mance records on monocular & multi-view depth estimation*

*as well as relative pose estimation. In summary, DUSt3R*

*makes many geometric 3D vision tasks easy. Code and mod*

*els at* *https://github.com/naver/dust3r**.*

**1. Introduction**

Unconstrained dense 3D reconstruction from multiple RGB

images is one long-researched end-goal of computer vi

sion [21, 58, 72]. In a nutshell, it is the task of estimat

ing the 3D geometry and camera parameters of a particular

scene, given a set of photographs of this scene. Not only

does it have numerous applications like mapping [12, 59],

navigation [13], archaeology [70, 99], cultural heritage

preservation [37], robotics [63], but perhaps more impor

tantly, it holds a fundamentally special place among all the

tasks from the 3D vision research field. Indeed, it sub

sumes nearly all geometric 3D vision tasks, and modern

approaches for 3D reconstruction thus consists in a sequen

tial succession of many components, such as keypoint de

tection [23, 26, 53, 77] and matching [9, 51, 81, 92], ro

bust estimation [3, 9, 137], Structure-from-Motion (SfM)

and Bundle Adjustment (BA) [18, 50, 83], dense Multi

View Stereo (MVS) [84, 103, 119, 134], etc. This rather

complex chain is of course a viable solution in some set

tings [30, 57, 61, 106, 110, 112, 123], yet we argue it is

quite unsatisfactory: each task is not solved perfectly and

adds noise to the next step, increasing the complexity and

the engineering effort required for the pipeline to work as a

whole. The absence of communication between each compo

nent is also quite telling: it would seem more reasonable if

they helped each other, *i.e*. dense reconstruction should nat

urally benefit from the sparse scene that was built to recover

camera poses, and vice-versa. On top of that, key steps in

this pipeline are brittle and prone to break in many cases [50].

For instance, the crucial stage of SfM that serves to estimate

all camera parameters is typically known to fail in many

common situations, *e.g*. when the number of scene views is

low [85], for objects with non-Lambertian surfaces [14], in

case of insufficient or overly large camera motion [12], etc.

In this paper, we present **DUSt3R**, a radically novel ap

proach for Dense Unconstrained Stereo 3D Reconstruction

from un-calibrated and un-posed cameras. The main compo

nent is a network that can regress a dense and accurate scene

representation solely from a *pair* of images, without prior

information regarding the scene nor the cameras (not even

the intrinsic parameters). The resulting scene representation

is based on *3D pointmaps* with rich properties: they simulta

neously encapsulate (a) the scene geometry, (b) the relation

between pixels and scene points and (c) the relation between

the two viewpoints. From this output alone, practically all

scene parameters (*i.e*. cameras and scene geometry) can be

straightforwardly recovered. This is possible because the

network jointly processes the input images and the resulting

3D pointmaps, thus learning to associate 2D patterns with

3D shapes and having the opportunities of solving multi

ple tasks simultaneously, enabling internal ‘collaboration’

between them.

Our model is trained in a fully-supervised manner using

a simple regression loss, leveraging large public datasets for

which ground-truth annotations are either synthetically gen

erated [56, 82], reconstructed from SfM softwares [47, 122]

or captured using dedicated sensors [22, 75, 94, 126]. We

drift away from the trend of integrating task-specific modules

[125], and instead adopt a fully data-driven strategy based

on a generic transformer architecture, not enforcing any geo

metric constraints at inference, but being able to benefit from

powerful pretraining schemes [114]. The network learns

strong geometric and shape priors, which is reminiscent of

those commonly leveraged in MVS, like shape from texture,

shading or contours [87].

To fuse predictions from multiple images pairs, we revisit

bundle adjustment (BA) for the case of pointmaps, hereby

achieving full-scale MVS. We introduce a global alignment

procedure that, contrary to BA, does not involve minimiz

ing reprojection errors. Instead, we optimize the camera

poses and the scene geometry directly in 3D space, which

is fast and shows excellent convergence in practice. Our

experiments show that the reconstructions are accurate and

consistent between views in real-life scenarios with various

unknown sensors. We further demonstrate that the same

architecture can handle *real-life* monocular and multi-view

reconstruction scenarios seamlessly. Examples of reconstruc

tions are shown in Fig. 1 and in the accompanying video.

In summary, our contributions are fourfold. First, we

present the first *holistic end-to-end 3D reconstruction*

*pipeline* from un-calibrated and un-posed images, that uni

fies monocular and binocular 3D reconstruction. Second, we

introduce the pointmap representation for MVS applications,

that enables the network to predict the 3D shape in a canoni

cal frame, while preserving the implicit relationship between

pixels and the scene. This effectively drops many constraints

of the usual perspective camera formulation. Third, we intro

duce an optimization procedure to globally align pointmaps

in the context of multi-view 3D reconstruction. Our proce

dure can extract effortlessly all usual intermediary outputs

of the classical SfM and MVS pipelines. In a sense, our ap

proach unifies all 3D vision tasks and considerably simplifies

over the traditional reconstruction pipeline, making DUSt3R

seem simple and easy in comparison. Fourth, we demon

strate promising performance on a range of 3D vision tasks

In particular, our all-in-one model achieves state-of-the-art

results on monocular and multi-view depth benchmarks, as

well as multi-view camera pose estimation.

**2. Related Work**

For the sake of space, we summarize here the most related

works in 3D vision, and refer the reader to Sec. B of the

supplementary for a more comprehensive review.

**Structure-from-Motion (SfM)** [18, 19, 40, 42, 83] aims

at reconstructing sparse 3D maps while jointly determin

ing camera parameters from a set of images. The tradi

tional pipeline starts from pixel correspondences obtained

from keypoint matching [4, 5, 39, 53, 80] between multiple

images to determine geometric relationships, followed by

bundle adjustment to optimize 3D coordinates and camera

parameters jointly. Recently, the SfM pipeline has under

gone substantial enhancements, particularly with the incor

poration of learning-based techniques into its subprocesses.

These improvements encompass advanced feature descrip

tion [23, 26, 77, 101, 127], more accurate image match

ing [3, 15, 27, 28, 51, 65, 81, 92, 96, 107], featuremetric

refinement [50], and neural bundle adjustment [49, 116]. De

spite these advancements, the sequential structure of the SfM

pipeline persists, making it vulnerable to noise and errors in

each individual component.

**MultiView Stereo (MVS)** is the task of densely reconstruct

ing visible surfaces, which is achieved via triangulation be

tween multiple viewpoints. In the classical formulation of

MVS, all camera parameters are supposed to be provided

as inputs. The fully handcrafted [31, 33, 84, 111, 133], the

more recent scene optimization based [30, 57, 60, 61, 106,

110, 112, 123], or learning based [45, 55, 69, 121, 124, 136]

approaches all depend on camera parameter estimates ob

tained via complex calibration procedures, either during the

data acquisition [1, 20, 85, 126] or using Structure-from

Motion approaches [42, 83] for in-the-wild reconstructions.

Yet, in real-life scenarios, the inaccuracy of pre-estimated

camera parameters can be detrimental for these algorithms

to work properly [78]. In this work, we propose instead to

directly predict the geometry of visible surfaces without any

explicit knowledge of the camera parameters.

**Direct RGB-to-3D**. Recently, some approaches aiming at di

rectly predicting 3D geometry from one or two RGB images

have been proposed. Since the problem is by nature ill-posed

without introducing additional assumptions, these methods

leverage neural networks that learn strong 3D priors from

large datasets to solve for ambiguities. These methods can

be classified into two groups. The first group leverages class

level object priors [66–68] or diffusion models to generate

novel views for object-centric reconstruction [52]. A sec

ond group of works, closest to our method, focuses instead

on general scenes. When starting from a single image, an

extensive usage of monocular depth estimation networks is

made [6, 73, 129, 131]. Depthmaps indeed encode a form

of 3D information and, combined with camera intrinsics,

can straightforwardly yield pixel-aligned 3D point-clouds.

SynSin [115], for example, performs new viewpoint syn

thesis from a single image by rendering feature-augmented

depthmaps knowing all camera parameters. If unknown,

camera intrinsics can be recovered by exploiting temporal

consistency in video frames [35, 90, 117] or regressed by

a specialized network [128, 130]. All these methods are,

however, intrinsically limited by the quality of depth esti

mates, which arguably is ill-posed for monocular settings.

To solve this issue, multi-view networks for direct 3D recon

struction like DeMon and DeepV2D have been proposed in

the past [98, 102, 139]. They are essentially based on the

idea of building a differentiable SfM pipeline, replicating

the traditional pipeline but training it end-to-end. As before,

however, ground-truth camera intrinsics are required as in

put, and the output is generally a depthmap and a relative

camera pose [102, 139]. In contrast, our network outputs

pointmaps, *i.e*. dense 2D fields of 3D points, which han

dle camera poses implicitly without requiring any camera

intrinsic parameters.

**Pointmaps.** Using a collection of pointmaps as shape rep

resentation is quite counter-intuitive for MVS, but its usage

is widespread for Visual Localization tasks, either in scene

dependent optimization approaches [7, 8, 10, 24, 46, 108,

109] or scene-agnostic inference methods [76, 95, 120]. Sim

ilarly, view-wise modeling is a common theme in monocular

3D reconstruction works [48, 88, 97, 105] and in view syn

thesis works [115], the idea being to store the canonical 3D

shape in multiple canonical views to work in image space.

These approaches usually leverage explicit perspective cam

era geometry, via rendering of the canonical representation.

**3. Method**

Before delving into the details of our method, we introduce

below some essential concepts.

**Pointmap**. In the following, we denote a dense 2D field

of 3D points as a *pointmap* *X* *∈* R*W**×**H**×*3 . In association

with its corresponding RGB image *I* of resolution *W* *×* *H*,

*X* forms a one-to-one mapping between image pixels and

3D scene points, *i.e*. *I**i,j* *↔* *X**i,j* , for all pixel coordinates

(*i, j*) *∈ {*1 *. . . W**} × {*1 *. . . H**}*. We assume here that each

camera ray hits a single 3D point, *i.e*. ignoring the case of

translucent surfaces.

**Cameras and scene**. Given camera intrinsics *K* *∈* R 3*×*3 ,

the pointmap *X* of the observed scene can be straight

forwardly obtained from the ground-truth depthmap *D* *∈*

R*W**×**H* as *X**i,j* = *K**−*1*D**i,j* [*i, j,* 1]*⊤* . Here, *X* is expressed

in the camera coordinate frame. In the following, we de

note as *X**n,m* the pointmap *X**n* from camera *n* expressed in

camera *m*’s coordinate frame:

 *\X {n}{m} = P_m P_n^{-1} h\left ( X^n \right ) \label {eq:pointmap} \vspace {-2mm}* 

(1)

where *P**m**, P**n* *∈* R 3*×*4 are the world-to-camera poses for

images *m* and *n*, and *h* : (*x, y, z*) *→* (*x, y, z,* 1) is the

homogeneous mapping.

**3.1. Overview**

We wish to build a network that solves the 3D reconstruction

task for the generalized stereo case through direct regression.

To that aim, we train a network *f* that takes as input two RGB

images *I* 1 *, I*2 *∈* R*W**×**H**×*3 and outputs two corresponding

pointmaps *X*1*,*1 *, X*2*,*1 *∈* R*W**×**H**×*3 with associated confi

dence maps *C* 1*,*1 *, C*2*,*1 *∈* R*W**×**H*. Note that both pointmaps

are expressed in the *same* coordinate frame of *I* 1 , which

radically differs from existing approaches but offers key ad

vantages (see Secs. 1, 2, 3.3 and 3.4). For the sake of clarity

and without loss of generalization, we assume here that both

images have the same resolution of *W* *×* *H*, but naturally in

practice their resolution can differ.

**Network architecture.** The architecture of our network *f*

is inspired by CroCo [114], making it straightforward to

heavily benefit from CroCo pretraining [113]. As shown in

Fig. 2, it is composed of two identical branches (one for each

image) comprising each an image encoder, a decoder and

a regression head. The two input images are first encoded

in a Siamese manner by the same weight-sharing ViT en

coder [25], yielding two token representations *F* 1 and *F* 2 :

The network then reasons over both of them jointly in the

decoder. Similarly to CroCo [114], the decoder is a generic

transformer network equipped with cross attention. Each

decoder block thus sequentially performs self-attention (each

token of a view attends to tokens of the same view), then

cross-attention (each token of a view attends to all other

tokens of the other view), and finally feeds tokens to a MLP.

Importantly, information is constantly shared between the

two branches during the decoder pass. This is crucial in

order to output properly aligned pointmaps. Namely, each

decoder block attends to tokens from the other branch:

*G* 1 *i* = DecoderBlock1 *i*

**3.3. Downstream Applications**

The rich properties of the output pointmaps allows us to

perform various convenient operations with relative ease.

**Point matching.** Establishing correspondences between pix

els of two images can be trivially achieved by nearest neigh

bor (NN) search in the 3D pointmap space. To minimize

errors, we typically retain reciprocal (mutual) correspon

dences *M*1

*,*

2 between images *I* 1 and *I* 2 , *i.e*. we have:

*M*1

*,*

2 = *{*(*a, b*) *|* *a* = NN1*,*2 (*b*) and *b* = NN2*,*1 (*a*)*}*

with NN*n,m*(*a*) = arg min

*b**∈{*0*,...,WH**}*





*X**b n,*1 *−* *X**a m,*1



*.*

**Recovering intrinsics.** By definition, the pointmap *X*1*,*1 is

expressed in *I* 1 ’s coordinate frame. It is therefore possible

to estimate the camera intrinsic parameters by solving a

simple optimization problem. In this work, we assume that

the principal point is approximately centered and pixels are

squares, hence only the focal *f*1 *∗* remains to be estimated:

 *f_1^\* = \argmin _{f_1} \sum _{i=0}^{W} \sum _{j=0}^{H} \C {1}{1}_{i,j} \left \Vert (i',j') - f_1 \frac {(\X {1}{1}_{i,j,0},\X {1}{1}_{i,j,1})}{\X {1}{1}_{i,j,2}} \right \Vert , \vspace {-1mm}* 

with *i* *′* = *i* *−* *W* 2 and *j* *′* = *j* *−* *H* 2 . Fast iterative solvers, *e.g*.

based on the Weiszfeld algorithm [71], can find the optimal

*f*1 *∗* in a few iterations. For the focal *f*2 *∗* of the second camera,

the simplest option is to perform the inference for the pair

(*I* 2 *, I*1 ) and use above formula with *X*2*,*2 instead of *X*1*,*1 .

**Relative pose estimation** can be achieved in several fashions.

One way is to perform 2D matching and recover intrinsics

as described above, then estimate the Epipolar matrix and

recover the relative pose [40]. Another, more direct way is

to compare the pointmaps *X*1*,*1 *↔* *X*1*,*2 (or, equivalently,

*X*2*,*2 *↔* *X*1*,*2 ) using Procrustes alignment [54] to get the

scaled relative pose *P* *∗* = *σ* *∗* [*R**∗* *|**t* *∗* ]:

 *P^\* = \argmin _{\sigma ,R,t} \sum _{i} \C {1}{1}_i \C {1}{2}_i \left \Vert \sigma (R \X {1}{1}_i + t) - \X {1}{2}_i \right \Vert ^2, \vspace {-2mm}* 

which can be achieved in closed-form. Procrustes alignment

is, unfortunately, sensitive to noise and outliers. A more

robust solution is to rely on RANSAC [29] with PnP [40, 44].

**Absolute pose estimation**, also termed visual localization,

can likewise be achieved in several different ways. Let *I* *Q*

denote the query image and *I* *B* the reference image for

which 2D-3D correspondences are available. First, intrinsics

for *I* *Q* can be estimated from *X**Q,Q* as explained above.

Then, one possibility is to run PnP-RANSAC [29, 44] from

2D pixel correspondences obtained between *I* *Q* and some

*I* *B*, which in turn yields 2D-3D correspondences for *I* *Q*.

Another solution is to get the relative pose between *I* *Q* and

*I* *B* as described previously. Then, we convert this pose to

world coordinate by scaling it appropriately, according to the

scale between *X**B,B* and the ground-truth pointmap for *I* *B*.

**3.4. Global Alignment**

The network *f* presented so far can only handle a pair of

images. We now present a fast and simple post-processing

optimization for larger scenes. It enables the alignment of

pointmaps predicted from multiple images into a joint 3D

space. This is possible thanks to the rich content of our

pointmaps, which encompasses by design two aligned point

clouds and their corresponding pixel-to-3D mapping.

**Pairwise graph.** Given a set of images *{**I* 1 *, I*2 *, . . . , I**N* *}*

for a given scene, we first construct a connectivity graph

*G* 

= (

*V*

*,* *E*) 

where 

*N* 

images form vertices 

*V* 

and each edge

*e* 

= (

*n, m*

) 

*∈ E* 

indicates that images 

*I*

*n* 

and 

*I* *m* share

some visual content. To that aim, we either use existing

off-the-shelf image retrieval methods, or we pass all pairs

through network *f* (inference takes *≈*25ms on a H100 GPU)

to measure their overlap from the average confidence in both

pairs, and then filter out low-confidence pairs.

**Global optimization.** We use the connectivity graph *G*

to recover *globally aligned* pointmaps *{**χ* *n* *∈* R*W**×**H**×*3*}*

for all cameras *n* = 1 *. . . N*. To that aim, we first pre

dict, for each image pair *e* = (*n, m*) *∈ E*, the pair

wise pointmaps *X**n,n**, X**m,n* and their associated confidence

maps *C* *n,n**, C**m,n*. For the sake of clarity, let us define

*X**n,e* := *X**n,n* and *X**m,e* := *X**m,n*. Since our goal involves

to express all pairwise predictions in a common coordinate

frame, we introduce a pairwise pose *P**e* *∈* R 3*×*4 and scaling

*σ**e* *>* 0 associated to each pair *e* *∈ E*. We then formulate the

following optimization problem:

 *\chi ^\* = \argmin _{\chi ,P,\sigma } \sum _{e \in \E } \sum _{v \in e} \sum _{i=1}^{HW} \C {v}{e}_i \left \Vert \chi _i^v - \sigma _e P_e \X {v}{e}_i \right \Vert . \label {eq:pose_optim} \vspace {-2mm}* (5)

Here, with some abuse of notation, we write *v* *∈* *e* for *v* *∈*

*{**n, m**}* if *e* = (*n, m*). The idea is that, for a given pair *e*, the

*same* rigid transformation *P**e* should align both pointmaps

*X**n,e* and *X**m,e* with the world-coordinate pointmaps *χ* *n* and

*χ* *m*, since *X**n,e* and *X**m,e* are by definition both expressed

in the same coordinate frame. To avoid the trivial optimum

where *σ**e* = 0*,* *∀**e* *∈ E*, we enforce that Q 

*e*

*σ**e* = 1.

**Recovering camera parameters.** A straightforward exten

sion to this framework enables to recover all cameras parame

ters. By simply replacing *χ* *n i,j* := *P**n* *−*1*h*(*K**n* *−*1*D**i,j n* [*i, j,* 1]*⊤*)

(*i.e*. enforcing a standard camera pinhole model as in Eq. (1)),

we can thus estimate all camera poses *{**P**n**}*, associated in

trinsics *{**K**n**}* and depthmaps *{**D**n**}* for *n* = 1 *. . . N*. To

accelerate convergence, we initialize all parameters using

pairwise relative pose estimates propagated along a maxi

mum spanning tree of *G*, see Sec. G of the supplementary.

**Discussion.** We point out that, contrary to traditional bundle

adjustment, this global optimization is fast and simple to

perform in practice. Indeed, we are not minimizing 2D

reprojection errors, as bundle adjustment normally does, but

3D projection errors. The optimization is carried out using

standard gradient descent and typically converges after a few

hundred steps, requiring mere seconds on a standard GPU.

**4. Experiments with DUSt3R**

**Training data**. We train our network with a mixture

of eight datasets: Habitat [82], MegaDepth [47], ARK

itScenes [22], Static Scenes 3D [56], Blended MVS [122],

ScanNet++ [126], CO3D-v2 [75] and Waymo [94]. These

datasets feature diverse scene types: indoor, outdoor, land

marks, synthetic, real-world, object-centric, etc. When im

age pairs are not directly provided with the dataset, we ex

tract them based on the method described in [113]. Specifi

cally, we utilize off-the-shelf image retrieval and point match

ing algorithms to match and verify image pairs. All in all,

we extract 8.5M pairs in total.

**Training details**. During each epoch, we randomly sam

ple an equal number of pairs from each dataset to compen

sate disparities in dataset sizes. We wish to feed relatively

high-resolution images to our network, say 512 pixels in

the largest dimension. To mitigate the high cost associated

with such input, we train our network sequentially, first on

224*×*224 images and then on larger 512-pixel images. We

randomly select the image aspect ratios for each batch (*e.g*.

16/9, 4/3, etc), so that at test time our network is familiar

with different image shapes. We crop images to the desired

aspect-ratio, and resize the largest dimension to 512 pixels.

We use standard data augmentation techniques and train

ing set-up overall. Our network architecture comprises a ViT

Large for the encoder [25], a ViT-Base for the decoder, both

with patches of 16*×*16 pixels, and a DPT head [73]. We refer

to the supplementary in Sec. H for more details on the train

ing and architecture. Before training, we initialize our net

work with the weights of an off-the-shelf CroCo pretrained

model [114]. Cross-View completion (CroCo) is a recently

proposed pretraining paradigm inspired by MAE [41] that

has been shown to excel on various downstream 3D vision

tasks [113], and is thus particularly suited to our framework.

**Evaluation**. In the remainder of this section, we bench

mark DUSt3R on a representative set of classical 3D vision

tasks, each time specifying datasets, metrics and compar

ing performance with existing state-of-the-art approaches.

We emphasize that all results are obtained with the *same*

DUSt3R model (our default model is denoted as ‘DUSt3R

512’, other DUSt3R models serve for the ablations in Sec.

F of the suppl.), *i.e*. we never finetune our model on a par

ticular downstream task (zero-shot settings). During test, all

images are rescaled to 512 pixels while preserving their as

pect ratio. Since there may exist different ‘routes’ to extract

task-specific outputs from DUSt3R, as described in Sec. 3.3

and Sec. 3.4, we precise each time the employed method.

**Recovering intrinsics** with DUSt3R is possible in monocu

lar and binocular settings, see Sec. E of the supplementary.

**Qualitative results**. As shown in Fig. 1, DUSt3R yields

high-quality dense 3D reconstructions even in challenging

situations. It can even *reconstruct scenes for which images*

*share no visual overlap* (top-right office example). We refer

the reader to the supplementary in Sec. A for more visual

izations of pairwise and multi-view reconstructions.

**4.1. Map-free Visual Localization**

**Dataset.** 

We experiment with the Map-free relocalization

benchmark [2], an extremely challenging test-bed were the

goal is to localize the camera in metric space given a single

reference image (*i.e*. without any map). The benchmark

comprises a training set which we do not use at all, 65 vali

dation and 130 test scenes. For each scene, the pose of every

frame in a video clip must be independently estimated w.r.t.

a single reference image. The video clip is captured with

a different device at a different moment (possibly months

before or after the reference image), and the ground-truth is

privately held-out, making the benchmark as fair as possible.

**Protocol.** The evaluation returns absolute camera pose ac

curacy (localization thresholds of 5 *◦* , 25cm) and Virtual

Correspondence Reprojection Error (VCRE) measured as

the average Euclidean distance of the reprojection errors of

virtual 3D points projected according to ground truth and

estimated camera poses. To evaluate DUSt3R, we first ex

tract pixel correspondences as described in Section 3.3 of the

main paper, then we estimate the relative camera pose using

RANSAC via the essential matrix using the provided bench

mark code. To find the metric scale of the scene, we leverage

metric depth from an off-the-shelf DPT-KITTI again us

ing the provided code, similarly to most other methods like

RoMa [28], LoFTR [92] and SuperPoint-SuperGlue [23, 81].

**Results.** Comparisons with the state of the art on the pri

vately held-out test set are reported in Tab. 1. Overall,

DUSt3R outperforms all state-of-the-art approaches, some

times by a large margin, achieving less than 1 meter of

median translation error, whereas other approaches usually

achieve between 1.5 and 2.5 meters in median translation

error. In terms of reprojection error, DUSt3R achieves more

than 50% precision at 90 pixel threshold and almost 70% in

AUC, which is again far better than most other approaches,

including RoMa [28] which relies on the powerful DINOv2

pretraining [62]. It thus appears that correspondences output

by DUSt3R are more robust than ones byexisting matching

methods, even though these methods are explicitly designed

and trained for matching, whereas DUSt3R is not. Indeed,

we point out that pixel correspondences are only one of many

by-products of our proposed reconstruction framework.

**4.2. Multi-view Pose Estimation**

We evaluate DUSt3R for the task of multi-view relative pose

estimation, with and without global alignment (Sec. 3.4).

**Datasets.** Following [104], we use two multi-view datasets,

CO3Dv2 [75] and RealEstate10k [140] for the evaluation.

CO3Dv2 contains 6 million frames extracted from approxi

mately 37k videos, covering 51 MS-COCO categories. The

ground-truth camera poses are annotated using COLMAP

from 200 frames in each video. RealEstate10k is an in

door/outdoor dataset with 10 million frames from about 80K

video clips on YouTube, the camera poses being obtained

by SLAM with bundle adjustment. We follow the protocol

introduced in [104] to evaluate DUSt3R on 41 categories

from CO3Dv2 and 1.8K video clips from the test set of

RealEstate10k. For each sequence, we random select 10

frames and feed all possible 45 pairs to DUSt3R.

**Baselines and metrics.** We compare DUSt3R pose es

timation results, obtained either from PnP-RANSAC or

global alignment, against the learning-based RelPose [135],

PoseReg [104] and PoseDiffusion [104], and structure-based

PixSFM [50], COLMAP+SPSG (COLMAP [84] extended

with SuperPoint [23] and SuperGlue [81]). Similar to [104],

we report the Relative Rotation Accuracy (RRA) and Rel

ative Translation Accuracy (RTA) for each image pair to

evaluate the relative pose error and select a threshold *τ* = 15

to report RTA@15 and RRA@15. Additionally, we calcu

late the mean Average Accuracy (mAA)@30, defined as the

area under the curve accuracy of the angular differences at

*min*(RRA@30*,* RTA@30).

**Results.** As shown in Table 2, DUSt3R with global align

ment (GA) achieves the best overall performance on the

two datasets and significantly outperforms the state-of-the

art PoseDiffusion [104]. Moreover, DUSt3R with PnP also

demonstrates superior performance over both learning and

structure-based existing methods. It is worth noting that

RealEstate10K results reported for PoseDiffusion are from

the model trained on CO3Dv2. Nevertheless, we assert that

our comparison is justified considering that RealEstate10K

is not used either during DUSt3R’s training. We also report

performance with less input views (between 3 and 10) in the

supplementary (Sec. C), in which case DUSt3R also yields

excellent performance on both benchmarks.

**4.3. Monocular Depth**

For this monocular task, we simply feed the same input im

age *I* to the network as *f*(*I, I*). By design, depth prediction

is simply the *z* coordinate in the predicted 3D pointmap.

**Datasets and metrics.** 

We benchmark DUSt3R on

two outdoor (DDAD [38], KITTI [34]) and three indoor

(NYUv2 [89], BONN [64], TUM [91]) datasets. We com

pare DUSt3R ’s performance to state-of-the-art methods

categorized in supervised, self-supervised and zero-shot set

tings, this last category corresponding to DUSt3R. We use

two metrics commonly used for monocular depth evalua

tions [6, 90]: the absolute relative error *AbsRel* between

target *y* and prediction *y*ˆ, *AbsRel* = *|**y* *−* *y*ˆ*|**/y*, and the pre

diction threshold accuracy, *δ*1*.*25 = max(ˆ*y/y, y/y*ˆ) *<* 1*.*25.

**Results.** In zero-shot setting, the state of the art is rep

resented by the recent SlowTv [90]. This approach col

lected a large mixture of curated datasets with urban, natu

ral, synthetic and indoor scenes, and trained one common

model. For every dataset in the mixture, camera parameters

are known or estimated with COLMAP. As Table 2 shows,

DUSt3R adapts well to outdoor and indoor environments.

It outperforms the self-supervised baselines [6, 36, 93] and

performs on-par with SoTA supervised baselines [73, 132].

**4.4. Multi-view Depth**

We evaluate DUSt3R for the task of multi-view stereo

depth estimation. Likewise, we extract depthmaps as the

*z*-coordinate of predicted pointmaps. In the case where mul

tiple depthmaps are available for the same image, we rescale

all predictions to align them together and aggregate all pre

dictions via a simple averaging weighted by the confidence.

**Datasets and metrics.** Following [86], we evaluate it on

the DTU [1], ETH3D [85], Tanks and Temples [43], and

ScanNet [20] datasets. We report the Absolute Relative

Error (rel) and Inlier Ratio (*τ* ) with a threshold of 1.03 on

each test set, and the averages across all test sets. Note that

we do not leverage the *ground-truth* camera parameters and

poses nor the *ground-truth* depth ranges, so our predictions

are only valid up to a scale factor. In order to perform

quantitative measurements, we thus normalize predictions

using the medians of the predicted depths and the ground

truth ones, as advocated in [86].

**Results.** We observe in Tab. 3 (left) that DUSt3R achieves

state-of-the-art accuracy on ETH-3D and outperforms most

recent state-of-the-art methods overall, even those using

ground-truth camera poses. Time-wise, our approach is also

much faster than the traditional COLMAP pipeline [83, 84].

This showcases the applicability of our method on a large

variety of domains, either indoors, outdoors, small scale or

large scale scenes, while not having been trained on the test

domains, except for the ScanNet test set, since the train split

is part of our Habitat dataset. We additionally provide the

comparison with other baselines in Tab. 7 of supplementary.

**4.5. 3D Reconstruction**

Finally, we measure the quality of our full reconstructions

obtained after the global alignment procedure described

in Sec. 3.4. We again emphasize that our method is the first

one to enable global unconstrained MVS, in the sense that we

have no prior knowledge regarding the camera parameters.

In order to quantify the quality of our reconstructions, we

simply align the predictions to the ground-truth coordinate

system. This is done by fixing the parameters as constants

in Eq. (5). This leads to consistent 3D reconstructions ex

pressed in the coordinate system of the ground-truth.

**Datasets and metrics.** We evaluate our predictions on the

DTU [1] dataset. We apply our network in a zero-shot setting,

*i.e*. we do not finetune on the DTU train set and apply our

model as is. In Tab. 3 (right) we report the averaged accuracy,

averaged completeness and overall averaged error metrics as

provided by the authors of the benchmarks. The accuracy for

a point of the reconstructed shape is defined as the smallest

Euclidean distance to the ground-truth, and the completeness

of a point of the ground-truth as the smallest Euclidean

distance to the reconstructed shape. The overall is simply

the mean of both previous metrics.

**Results.** Our method does not reach the accuracy levels of

the best methods. In our defense, these methods all lever

age GT poses and train specifically on the DTU train set

whenever applicable. Furthermore, best results on this task

are usually obtained via sub-pixel accurate triangulation, re

quiring the use of explicit camera parameters, whereas our

approach relies on regression, which is known to be less ac

curate. Yet, without prior knowledge about the cameras, we

reach an average accuracy of 2*.*7*mm*, with a completeness

of 0*.*8*mm*, for an overall average distance of 1*.*7*mm*. We

believe this level of accuracy to be of great use in practice,

considering the *plug-and-play* nature of our approach.

**5. Conclusion**

We presented a novel paradigm to solve not only 3D recon

struction in-the-wild without prior information about scene

nor cameras, but a whole variety of 3D vision tasks as well.

